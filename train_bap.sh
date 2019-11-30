

python train_bap.py train\
    --model-name inception \
    --batch-size 12 \
    --dataset bird \
    --image-size 512 \
    --input-size 448 \
    --checkpoint-path checkpoint/bird \
    --optim sgd \
    --scheduler step \
    --lr 0.001 \
    --momentum 0.9 \
    --weight-decay 1e-5 \
    --workers 4 \
    --parts 32 \
    --epochs 80 \
    --use-gpu \
    --multi-gpu \
    --gpu-ids 0,1 \
