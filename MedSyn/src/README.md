# Training of MedSyn
This is a one-key running bash, which will run both low-res and high-res. But the training can be done independently
```bash
sh run_train.sh
```

# Running inference

```bash
sh run_inference.sh
```

The inference process requires a GPU with at least 32GB of memory. On a single NVIDIA Tesla V100 GPU, it takes about 4 minutes to generate an image.

Our checkpoint for pre-trained language model is available [here](https://www.dropbox.com/scl/fi/d6tg6si72nnjfa87vawsl/pretrained_lm.gz?rlkey=fcnyrmy1i3xi9frzjchc68kh3&st=gq6xofnh&dl=0).

Our checkpoint for diffusion model pre-trained on UPMC dataset is available [here](https://drive.google.com/file/d/1AAlEN_dB7C0aVMJ81mKBlYnSqMVOk-tl/) (Application required).