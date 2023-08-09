# Training

## Launch Redis

Launch in-memory database, only need once

```bash
# It launches redis at ports 39996-39999.
bash ./scripts/start_redis.sh
```

**NOTE**

- If **data or preprocessing** changed, run `pkill redis-server` before further experiments
- Try `--workers` from 8 to 32 for best performance
- First epoch after launch the server could be slow, but should be fast later
- Set `--redis_ports <ports>` according to your redis setup.

## Pre-training

```bash
cd Pretrain
bash run.sh
```

## Finetuning

- Prepare pretrained models: Copy the pretrained model to the `pretrained_models` directory in BTCV

```bash
cd WORD
bash run.sh
```
