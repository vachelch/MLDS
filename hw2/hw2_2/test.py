from dataset import Dataset 
from my_models import Encoder, BahdanauAttnDecoderRNN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import numpy as np 

import time
import random
import sys, os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 100
vocab_size = 8000
latent_dim = 512
hidden_size = 512
n_layers = 2
dropout = 0.1
decoder_learning_ratio = 5.0

learning_rate = 0.001
teacher_forcing_ratio = 1


min_seq_len = 4
max_seq_len = 26

encoder_path = 'encoder.pt'
decoder_path = 'decoder.pt'

input_file = sys.argv[1]
output_file = sys.argv[2]



dataset = Dataset(vocab_size = vocab_size, min_len = min_seq_len, max_len = max_seq_len)
dataset.prepare_data(load_local = True, train = False)

pad_idx = dataset.pad_idx
bos_idx = dataset.bos_idx
eos_idx = dataset.eos_idx

encoder = Encoder(
    vocab_size = vocab_size, 
    latent_dim = latent_dim, 
    hidden_size = hidden_size, 
    pad_idx = pad_idx, 
    n_layers = n_layers, 
    dropout = dropout
    )
decoder = BahdanauAttnDecoderRNN(
    vocab_size = vocab_size, 
    latent_dim = latent_dim, 
    hidden_size = hidden_size, 
    pad_idx = pad_idx, 
    n_layers = n_layers, 
    dropout = dropout
    )

encoder = encoder.cuda()
decoder = decoder.cuda()
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))


def predict(input_batches, input_lengths, encoder, decoder):
    encoder_hidden = Variable(torch.zeros(n_layers, batch_size, hidden_size).cuda())
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, encoder_hidden)

    #prepare decoder input
    decoder_input = Variable(torch.LongTensor([bos_idx] * batch_size).cuda())
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    res_idx = []
    i = 0
    while  i <= dataset.max_len:
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1, dim = 1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        i += 1
        res_idx.append(decoder_input.cpu().data.numpy())

    res_idx = np.array(res_idx).transpose(1,0)

    res = []
    for row in res_idx:
        words = []
        for idx in row:
            if idx == eos_idx or idx == pad_idx:  
                break
            else:
                words.append(dataset.idx2word[idx])
        res.append(words)

    return res


with open(input_file, 'r', encoding = 'utf8') as f:
    test_input = []
    for row in f.readlines():
        test_input.append(row.split())

test_input_idxs, test_input_lengths = dataset.sentens2idxs(test_input)

batch_number = test_input_idxs.size(1) // batch_size

preds = []

for i in range(batch_number):
    start = i * batch_size 
    end = (i+1) * batch_size

    input_batches = test_input_idxs[:, start:end]
    input_batches = Variable(input_batches.cuda())

    input_lengths = test_input_lengths[start:end]

    pred = predict(input_batches, input_lengths, encoder, decoder)
    for row in pred:
        preds.append(row)

with open(output_file, 'w', encoding = 'utf8') as f:
    for row in preds:
        f.write('{}\n'.format(' '.join(row)))
























