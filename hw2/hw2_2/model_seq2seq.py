from dataset import Dataset 
from my_models import Encoder, BahdanauAttnDecoderRNN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import time
import random

n_iters = 100000

batch_size = 100
vocab_size = 8000
latent_dim = 512
hidden_size = 512
n_layers = 2
dropout = 0.1
decoder_learning_ratio = 5.0

learning_rate = 0.0001
teacher_forcing_ratio = 0.9

data_path = "clr_conversation.txt"
min_seq_len = 4
max_seq_len = 26

encoder_path = 'encoder.pt'
decoder_path = 'decoder.pt'


dataset = Dataset(data_path = data_path, vocab_size = vocab_size, min_len = min_seq_len, max_len = max_seq_len)
dataset.prepare_data(load_local = False, train = False)

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
# encoder.load_state_dict(torch.load(encoder_path))

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

criterion = nn.CrossEntropyLoss()


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    batch_size = input_batches.size(1)
    encoder_hidden = Variable(torch.zeros(n_layers, batch_size, hidden_size).cuda())
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, encoder_hidden)

    #prepare decoder input
    decoder_input = Variable(torch.LongTensor([bos_idx] * batch_size).cuda())
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    max_target_length = max(target_lengths)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(max_target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_batches[i]

            loss += criterion(decoder_output, target_batches[i])

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(max_target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1, dim = 1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_batches[i])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data / (target_lengths.sum()*1.0)

def predict(input_batches, input_lengths, encoder, decoder):
    encoder_hidden = Variable(torch.zeros(n_layers, batch_size, hidden_size).cuda())
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, encoder_hidden)

    #prepare decoder input
    decoder_input = Variable(torch.LongTensor([bos_idx] * batch_size).cuda())
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    res = ''
    i = 0
    while decoder_input.data[0] != eos_idx and i < dataset.max_len:
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1, dim = 1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        if decoder_input.data[0] != eos_idx:
            res += dataset.idx2word[decoder_input.data[0]]
        i += 1
    return res

print_every = 500
loss = 0
dataset.data_generator(batch_size = batch_size)

start = time.time()
for i_iter in range(n_iters):
    input_batches, input_lengths, target_batches, target_lengths = dataset.next_batch()
    input_batches = Variable(input_batches.cuda())
    target_batches = Variable(target_batches.cuda())

    loss += train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    if i_iter % print_every == 0:
        loss_aver = loss / print_every
        end = time.time()
        print("{:.1f} s, iter:{}({:.2f}%), loss: {:.2f}".format(end-start, i_iter, i_iter/n_iters*100, loss.cpu()[0]))
        loss = 0
        start = time.time()

        res = predict(input_batches, input_lengths, encoder, decoder)
        ques = dataset.idxs2sentence(input_batches.cpu().data.numpy()[:, 0])
        ans = dataset.idxs2sentence(target_batches.cpu().data.numpy()[:, 0])
        print(">", ques)
        print("=", ans)
        print("<", res)


        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)







