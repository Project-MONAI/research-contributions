python -m torch.distributed.launch --nproc_per_node=8 --master_port=11223 main.py --batch_size=2 \
    --num_steps=30000 \
    --lrdecay \
    --eval_num=500 \
    --lr=5e-4 \
    --decay=0.1 \
    --norm_pix_loss \
    --redis_ports 39996 39997 39998 39999 \
    --redis_compression zlib
