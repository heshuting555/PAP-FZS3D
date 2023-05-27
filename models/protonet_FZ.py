""" Prototypical Network 


"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from models.attention import SelfAttention, QGPA
from models.gmmn import GMMNnetwork


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class ProtoNetAlignFZ(nn.Module):
    def __init__(self, args):
        super(ProtoNetAlignFZ, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = 'cosine'
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.use_align = args.use_align
        self.use_linear_proj = args.use_linear_proj
        self.use_supervise_prototype = args.use_supervise_prototype
        if args.use_high_dgcnn:
            self.encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        else:
            self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        if self.use_linear_proj:
            self.conv_1 = nn.Sequential(nn.Conv1d(args.train_dim, args.train_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.train_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.use_transformer = args.use_transformer
        if self.use_transformer:
            self.transformer = QGPA()
        if args.use_zero:
            self.generator = GMMNnetwork(args.noise_dim, args.noise_dim, args.train_dim, args.train_dim, args.gmm_dropout)

    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat, _ = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat, xyz = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)
        # prototype learning
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat)
        prototypes = [bg_prototype] + fg_prototypes

        self_regulize_loss = 0
        if self.use_supervise_prototype:
            self_regulize_loss = self.sup_regulize_Loss(prototypes, support_feat, fg_mask, bg_mask)

        if self.use_transformer:
            prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0).repeat(query_feat.shape[0], 1, 1)
            prototypes_all_post = self.transformer(query_feat, query_feat, prototypes_all)
            prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]
            query_pred = torch.stack(similarity, dim=1)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)
        else:
            similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
            query_pred = torch.stack(similarity, dim=1)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)
        align_loss = 0

        if self.use_align:
            align_loss_epi = self.alignLoss_trans(query_feat, query_pred, support_feat, fg_mask, bg_mask)
            align_loss += align_loss_epi

        prototypes_all = prototypes_all.clone().detach()
        return query_pred, loss + align_loss + self_regulize_loss, prototypes_all

    def forward_test_semantic(self, support_x, support_y, query_x, query_y, embeddings=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """

        query_feat, xyz = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)
        # prototype learning
        if self.use_transformer:
            prototypes_all = embeddings
            prototypes_all_post = self.transformer(query_feat, query_feat, prototypes_all)
            prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]
            query_pred = torch.stack(similarity, dim=1)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)

        return query_pred, loss

    def sup_regulize_Loss(self, prototype_supp, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype suppoort self alignment branch

        Args:
            prototypes: embedding features for query images
                expect shape: N x C x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            prototypes = [prototype_supp[0], prototype_supp[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2, xyz = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            if self.use_linear_proj:
                return self.conv_1(torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1)), xyz
            else:
                return torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1), xyz
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def calculateSimilarity_trans(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)

    def alignLoss_trans(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0)
                prototypes_all_post = self.transformer(img_fts, img_fts, prototypes_all)
                # prototypes_all_post = self.transformer(img_fts, qry_fts[0:1], prototypes_all)
                prototypes_new = [prototypes_all_post[0, 0], prototypes_all_post[0, 1]]

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes_new]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss
