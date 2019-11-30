

python train_bap.py test\
    --model-name inception \
    --batch-size 12 \
    --dataset bird \
    --image-size 512 \
    --input-size 448 \
    --checkpoint-path checkpoint/bird/model_best.pth.tar \
    --use-gpu \
    --multi-gpu \
    --gpu-ids 0,1 \
