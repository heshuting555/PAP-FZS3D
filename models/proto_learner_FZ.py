""" ProtoNet with/without attention learner for Few-shot 3D Point Cloud Semantic Segmentation


"""
import torch
from torch import optim
from torch.nn import functional as F

from models.protonet_FZ import ProtoNetAlignFZ
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
from models.gmmn import GMMNLoss
import numpy as np


class ProtoLearnerFZ(object):

    def __init__(self, args, mode='train'):

        self.model = ProtoNetAlignFZ(args)

        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        if mode == 'train':
            if args.use_attention:
                if args.use_transformer:
                    self.optimizer = torch.optim.Adam(
                    [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.transformer.parameters(), 'lr': args.trans_lr},
                     {'params': self.model.att_learner.parameters()}
                     ], lr=args.lr)
                else:
                    self.optimizer = torch.optim.Adam(
                    [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.att_learner.parameters()}
                     ], lr=args.lr)
            else:
                self.optimizer = torch.optim.Adam(
                    [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.linear_mapper.parameters()}], lr=args.lr)
            #set learning rate scheduler
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,
                                                          gamma=args.gamma)
            # load pretrained model for point cloud encoding
            self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
        elif mode == 'test':
            # Load model checkpoint
            self.model = load_model_checkpoint(self.model, args.model_checkpoint_path, mode='test')
        else:
            raise ValueError('Wrong GMMLearner mode (%s)! Option:train/test' %mode)
        self.optimizer_generator = torch.optim.Adam(self.model.generator.parameters(), lr=args.generator_lr)
        self.criterion_generator = GMMNLoss(
            sigma=[2, 5, 10, 20, 40, 80], cuda=True).build_loss()
        self.noise_dim = args.noise_dim
        self.gmm_weight = args.gmm_weight
        self.n_way = args.n_way
        vec_name = 'glove'
        if args.dataset == 's3dis':
            self.bg_id = 12
            self.embeddings = torch.from_numpy(np.load('dataloaders/S3DIS_{}.npy'.format(vec_name))) # S3DIS_fasttext.npy S3DIS_word2vec_google.npy
        elif args.dataset == 'scannet':
            self.bg_id = 0
            self.embeddings = torch.from_numpy(np.load('dataloaders/ScanNet_{}.npy'.format(vec_name))) # S3DIS_fasttext.npy S3DIS_word2vec_google.npy
        self.embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=1)

    def train(self, data, sampled_classes):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        [support_x, support_y, query_x, query_y] = data
        self.model.train()

        query_logits, loss, prototype_gts = self.model(support_x, support_y, query_x, query_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        self.optimizer_generator.zero_grad()

        support_embeddings = torch.cat([self.embeddings[self.bg_id].unsqueeze(0), self.embeddings[sampled_classes]], dim=0).to(query_pred.device)
        prototype_fakes = []

        for i in range(self.n_way):
            z_g = torch.rand((support_embeddings.shape[0], self.noise_dim)).cuda()
            prototype_fakes.append(self.model.generator(support_embeddings, z_g.float()))
        prototype_fakes = torch.stack(prototype_fakes, dim=0)
        g_loss_bg = self.criterion_generator(prototype_fakes[:, :1, :].flatten(0, 1), prototype_gts[:, :1, :].flatten(0, 1))
        g_loss_fg = self.criterion_generator(prototype_fakes[:, 1:, :].flatten(0, 1), prototype_gts[:, 1:, :].flatten(0, 1))
        g_loss = g_loss_bg * self.gmm_weight + g_loss_fg
        g_loss.backward()
        self.optimizer_generator.step()

        return loss, accuracy

    def test_semantic(self, data, sampled_classes):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [_, _, query_x, query_y] = data
        self.model.eval()

        support_embeddings = torch.cat([self.embeddings[self.bg_id].unsqueeze(0), self.embeddings[sampled_classes]], dim=0).cuda()

        with torch.no_grad():
            prototype_fakes = []
            for i in range(self.n_way):
                z_g = torch.rand((support_embeddings.shape[0], self.noise_dim)).cuda()
                prototype_fakes.append(self.model.generator(support_embeddings, z_g.float()))

            prototype_fakes = torch.stack(prototype_fakes, dim=0)

            logits, loss = self.model.forward_test_semantic(_, _, query_x, query_y, prototype_fakes)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0] * query_y.shape[1])

        return pred, loss, accuracy