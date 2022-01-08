# GNPassGAN

Repository for the submission review of paper "Creating Indistinguishable Honeywords With Improved Generative Adversarial Networks."  
GNPassGAN is an offline password guessing tool based on PassGAN with the implementation of Gradient Normalization in Pytorch 1.10.

The model from PassGAN is taken from [_Improved Training of Wasserstein GANs_](https://arxiv.org/abs/1704.00028) and it is a pytorch version of the  [improved_wgan_training](https://github.com/caogang/wgan-gp).

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
2) Train for 200,000 iterations, saving checkpoints every 5000.
3) Use the default hyperparameters from the paper for training.
```
python3 models.py --training-data data/YOUR TRAINING DATA --output-dir output
```
