import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np 
import math

class Encoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, hidden_size, pad_idx = 0, n_layers = 1, dropout = 0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, latent_dim, pad_idx)
        self.encoder = nn.GRU(latent_dim, hidden_size, n_layers, dropout = dropout, batch_first = True)


    def forward(self, encoder_inp, input_lengths, hidden = None):
        embedded = self.embedding(encoder_inp).transpose_(0, 1)

        sorted_index = np.argsort(-input_lengths)
        unsorted_index = np.argsort(sorted_index)

        sorted_index_tensor = Variable(torch.LongTensor(sorted_index).cuda())
        unsorted_index_tensor = Variable(torch.LongTensor(unsorted_index).cuda())

        sorted_outp = embedded[sorted_index_tensor]
        input_lengths = input_lengths[sorted_index]

        packed = pack_padded_sequence(sorted_outp, input_lengths, batch_first = True)
        outputs, hidden = self.encoder(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first = True)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        hidden.transpose_(0, 1)

        #unsort
        outputs = outputs[unsorted_index_tensor]
        hidden = hidden[unsorted_index_tensor]
        hidden.transpose_(0, 1)
        outputs.transpose_(0, 1)

        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies, dim = 1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, latent_dim, hidden_size, pad_idx = 0, n_layers=1, dropout=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        # Define layers
        self.embedding = nn.Embedding(vocab_size, latent_dim, pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + latent_dim, hidden_size, n_layers, dropout=dropout)
        #self.attn_combine = nn.Linear(hidden_size + latent_dim, hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).unsqueeze(0) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden.contiguous())
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim = 1)
        # output = F.log_softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden



class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, hidden_size, pad_idx = 0, n_layers = 1, dropout = 0.1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.output_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, latent_dim, pad_idx)
        self.gru = nn.GRU(latent_dim, hidden_size, n_layers,batch_first = True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).unsqueeze(1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden.contiguous()) #output(batch_size * 1 * hidden_size)
        output = output.squeeze(1)
        output = self.softmax(self.out(output)) #batch_size * vocab1_size
        return output, hidden




