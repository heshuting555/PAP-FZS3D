GPU_ID=1

DATASET='s3dis'
SPLIT=0
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis_PAP/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20
BASE_WIDTHS='[128, 64]'

PRETRAIN_CHECKPOINT='./log_s3dis/log_pretrain_s3dis_S0'
N_WAY=2
K_SHOT=1
N_QUESIES=1
N_TEST_EPISODES=100

NUM_ITERS=40000
EVAL_INTERVAL=2000
LR=0.001
DECAY_STEP=5000
DECAY_RATIO=0.5


args=(--phase 'prototrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --use_transformer --use_supervise_prototype
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT" --use_attention
      --use_align --use_high_dgcnn --pc_augm_shift 0.1
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --use_linear_proj
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS"
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES
      --trans_lr 1e-4
            )

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
