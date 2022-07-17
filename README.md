# GNPassGAN

This repository is created for the paper "GNPassGAN: Improved Generative Adversarial Networks For Trawling Offline Password Guessing"  published at ASSS 2022.

GNPassGAN is an offline password guessing tool based on PassGAN with the implementation of [Gradient Normalization](https://github.com/basiclab/GNGAN-PyTorch) in Pytorch 1.10.

The model used in PassGAN is inspired from paper [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) and its pytorch version implementation [improved_wgan_training](https://github.com/caogang/wgan-gp).

## Install dependencies

```bash
# requires CUDA 10 to be pre-installed
python3 -m venv .venv 
source .venv/bin/activate  
pip3 install -r requirements.txt
```
## Generate password samples
```bash
# generate 100,000,000 passwords
python3 sample.py \
	--input-dir output \
	--output generated/sample.txt \
  	--seq-length 12 \
  	--num-samples 10000000
```
## Train your own models

1) Prepare your own dataset for training first.
2) Train for 200,000 iterations, saving checkpoints every 10,000.
3) Use the default hyperparameters from the paper for training.
```
python3 models.py --training-data data/YOUR TRAINING DATA --output-dir output
```

## Check matching accuracy
```bash
# change data path
python3 accuracy.py \
	--input-generated generated/gnpassgan/10/180000iter/8.txt \
	--input-test data/test_rockyou10.txt
```

## Citation
If you find our work is relevant to your research, please cite:
```bash
@inproceedings{yugnpassgan,
author={Yu, Fangyi and Martin, Miguel Vargas},
booktitle={2022 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW)},
title={GNPassGAN: Improved Generative Adversarial Networks For Trawling Offline Password Guessing},
year={2022},
pages={10-18},
doi={10.1109/EuroSPW55150.2022.00009}}
