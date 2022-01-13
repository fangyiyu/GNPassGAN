# Reference: https://github.com/fangyiyu/GNGAN-PyTorch
import torch
import tflib as lib
import os, sys
sys.path.append(os.getcwd())
import time
import pickle
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tflib.plot
from sklearn.preprocessing import OneHotEncoder
import utils
import argparse
from torch.nn import BCEWithLogitsLoss

'''
python3 models.py --training-data data/rockyou_sorted_preprocessed.txt --output-dir output 

python3 sample.py \
	--input-dir output \
	--output generated/5n.txt \
    --num-samples 100000
'''
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=10000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 10000)')

    parser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=12,
                        dest='seq_length',
                        help='The maximum password length')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    
    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    return parser.parse_args()
args = parse_args()

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

lib.print_model_settings(locals().copy())
lines, charmap, inv_charmap = utils.load_dataset(
    path=args.training_data,
    max_length=args.seq_length)

# Pickle to avoid encoding errors with json
with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
    pickle.dump(charmap, f)

with open(os.path.join(args.output_dir, 'charmap_inv.pickle'), 'wb') as f:
    pickle.dump(inv_charmap, f)
    
print("Number of unique characters in dataset: {}".format(len(charmap)))

table = np.arange(len(charmap)).reshape(-1, 1)
one_hot = OneHotEncoder()
one_hot.fit(table)
# ==================Definition Start======================

class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(args.layer_dim, args.layer_dim, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(args.layer_dim, args.layer_dim, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(128, args.layer_dim*args.seq_length)
        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.conv1 = nn.Conv1d(args.layer_dim, len(charmap), 1)

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, args.layer_dim, args.seq_length) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(args.batch_size*args.seq_length, -1)
        output = torch.tanh(output)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))
        

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.conv1d = nn.Conv1d(len(charmap), args.layer_dim, 1)
        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        
        self.linear = nn.Linear(args.seq_length*args.layer_dim, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, args.seq_length*args.layer_dim)
        output = self.linear(output)
        return output

def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+args.batch_size]],
                dtype='int32'
            )

def normalize_gradient(netC, x):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    x: real_data_v
    f: C_real before mean
    
    """
    x.requires_grad_(True)
    f = netC(x)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)]) 
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat


def generate_samples(netG):
    noise = torch.randn(args.batch_size, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    # noisev = autograd.Variable(noise, volatile=True)
    with torch.no_grad():
            noisev = noise.cuda()
    samples = netG(noisev)
    samples = samples.view(-1, args.seq_length, len(charmap))
    # print samples.size()

    samples = samples.cpu().data.numpy()

    samples = np.argmax(samples, axis=2)
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(inv_charmap[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples
# ==================Definition End======================

netG = Generator()
netC = Critic()
print(netG)
print(netC)
loss_fn = BCEWithLogitsLoss()

if use_cuda:
    netC = netC.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerC = optim.Adam(netC.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

data = inf_train_gen()

true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[10*args.batch_size:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[:10*args.batch_size], tokenize=False) for i in range(4)]
for i in range(4):
    print ("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]

for iteration in range(args.iters + 1):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netC.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(args.critic_iters):
        _data = data.__next__()
        data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(args.batch_size, -1, len(charmap))
        #print data_one_hot.shape
        real_data = torch.Tensor(data_one_hot)
        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        netC.zero_grad()

        noise = torch.randn(args.batch_size, 128)
        if use_cuda:
            noise = noise.cuda(gpu)

        with torch.no_grad():
            noisev = noise.cuda()
        fake = autograd.Variable(netG(noisev).data)
        
        pred_real = normalize_gradient(netC, real_data_v)   # net_D(x_real)
        pred_fake = normalize_gradient(netC, fake)   # net_D(x_fake)


        loss_real = loss_fn(pred_real, torch.ones_like(pred_real))
        loss_fake_c = loss_fn(pred_fake, torch.zeros_like(pred_fake))
        loss = loss_real  + loss_fake_c 
        loss.backward() 
        optimizerC.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netC.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(args.batch_size, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    pred_fake = normalize_gradient(netC, fake)   # net_D(x_fake)
    loss_fake = loss_fn(pred_fake, torch.ones_like(pred_fake))
    loss_fake.backward() 
    optimizerG.step()


    lib.plot.output_dir = args.output_dir
    lib.plot.plot('time', time.time() - start_time)
    lib.plot.plot('train critic cost', loss.cpu().data.numpy())
    lib.plot.plot('train gen cost', loss_fake.cpu().data.numpy())

    if iteration % 100 == 0 and iteration > 0:

            samples = []
            for i in range(10):
                samples.extend(generate_samples(netG))

            for i in range(4):
                lm = utils.NgramLanguageModel(i+1, samples, tokenize=False)
                lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

            # with open(os.path.join(args.output_dir, 'samples', 'samples_{}.txt').format(iteration), 'w') as f:
            #     for s in samples:
            #         s = "".join(s)
            #         f.write(s + "\n")

    if iteration % args.save_every == 0 and iteration > 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('output/checkpoints/', iteration))
        torch.save(netC.state_dict(), '%s/netD_epoch_%d.pth' % ('output/checkpoints/', iteration))

    if iteration == args.iters:
        print("...Training done.")
        
    if iteration % 100 == 0:
        lib.plot.flush()

    lib.plot.tick()
        
# Time stamp
localtime = time.asctime(time.localtime(time.time()) )
print("End.")
print("Local current time :", localtime)

