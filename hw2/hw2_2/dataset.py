import re
from collections import Counter
import pickle
import numpy as np 
import torch

import random

unk_ratio = 0.1

class Dataset:
    def __init__(self, data_path = 'clr_conversation.txt', vocab_size = 5000, min_len = 4, max_len = 15):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def prepare_data(self, load_local = True, train = True):
        if load_local:
            self.word2idx = pickle.load(open('word2idx.pk', 'rb'))
            self.idx2word = pickle.load(open('idx2word.pk', 'rb'))
            if train:
                self.ques = np.load('ques.npy')
                self.ans = np.load('ans.npy')
        else:
            ques, ans, length = self.trim_data()
            self.build_dict(ques, ans)
            self.trim_data_2th(ques, ans, length)

    def trim_data(self):
        data = []
        with open(self.data_path, 'r', encoding = 'utf8') as f:
            for row in f.readlines():
                data.append(row.strip())
        
        length = len(data)
        print('Original data total rows:', length)

        ques = []
        ans = []

        for i in range(length-1):
            # ques_char = not bool(re.search('[a-zA-Z\d]', data[i]))
            # ans_char = not bool(re.search('[a-zA-Z\d]', data[i+1]))
            ques_row = data[i].split()
            ans_row = data[i+1].split()

            ques_true = (data[i] != '+++$+++' and self.min_len <= len(ques_row) <= self.max_len)
            ans_true = (data[i+1] != '+++$+++' and self.min_len <= len(ans_row) <= self.max_len)

            if ques_true and ans_true:
                ques.append(ques_row)
                ans.append(ans_row)
        print("first trimed data rows: {}({:.1f}%)".format(len(ques), len(ques)/length*100))

        return ques, ans, length


    def build_dict(self, ques, ans):
        word_count = Counter()
        for row in ques:
            for word in row:
                if word.isdigit():
                    word = '<NUM>'
                word_count[word] += 1

        self.word2idx = {'<PAD>':0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<NUM>': 4}
        self.idx2word = {0: '<PAD>', 1: '<BOS>',  2: '<EOS>', 3: '<UNK>', 4: '<NUM>'}

        idx = 4
        for pair in word_count.most_common(self.vocab_size-5):
            word, _ = pair
            self.word2idx[word] = idx
            self.idx2word[idx] = word

            idx += 1

        print('dictionary have been built, with size {}'.format(self.vocab_size))
        pickle.dump(self.word2idx, open('word2idx.pk', 'wb'))
        pickle.dump(self.idx2word, open('idx2word.pk', 'wb'))
        pickle.dump(word_count, open('sorted_word_count.pk', 'wb'))
        # word_count = pickle.load(open('sorted_word_count.pk', 'rb'))

    def trim_data_2th(self, ques_arr, ans_arr, length):
        #trim <UNK>
        ques = []
        ans = []

        for qrow, arow in zip(ques_arr, ans_arr):
            q_idxs = []
            a_idxs = []

            unk_count_q = 0
            for word in qrow:
                if word not in self.word2idx:
                    unk_count_q += 1

            unk_count_a = 0
            for word in arow:
                if word not in self.word2idx:
                    unk_count_a += 1
            
            if unk_count_a <= 1 and unk_count_q <= 1:
                add_unk = True if random.random() < unk_ratio else False
                if (unk_count_a == 0 and unk_count_q == 0) or add_unk:
                    ques.append(qrow)
                    ans.append(arow)

        
        self.ques = np.array(ques)
        self.ans = np.array(ans)

        print("training data(second trimed) rows:{}({:.1f}%)".format(len(self.ques), len(self.ques)/length*100))

        np.save('ques.npy', self.ques)
        np.save('ans.npy', self.ans)


    def sentence2idxs(self, sent):
        idxs = []
        for word in sent:
            if word in self.word2idx:
                idxs.append(self.word2idx[word])
            else:
                idxs.append(self.unk_idx)
        idxs.append(self.eos_idx)
        return idxs

    def pad_sentences(self, idxs, max_len):
        for row in idxs:
            length = len(row)
            for i in range(length, max_len):
                row.append(self.pad_idx)

    def idxs2sentence(self, idxs):
        res = ''
        for idx in idxs:
            if idx == self.eos_idx:
                break
            else:
                res += self.idx2word[idx]
        return res


    def batch(self, batch_size):
        while 1:
            lengths = len(self.ques)
            selected = np.random.randint(0, lengths, batch_size)

            ques_batch = self.ques[selected]
            ans_batch = self.ans[selected]

            input_batches, input_lengths = self.sentens2idxs(ques_batch)
            target_batches, target_lengths = self.sentens2idxs(ans_batch)

            yield (input_batches, input_lengths, target_batches, target_lengths)

    def data_generator(self, batch_size = 64):
        self.generator = self.batch(batch_size)

    def next_batch(self):
        return next(self.generator)

    
    def sentens2idxs(self, batch):
        data = []
        lengths = []

        for sentence in batch:
            data.append(self.sentence2idxs(sentence))
            lengths.append(len(sentence) + 1)

        max_len = max(lengths)
        self.pad_sentences(data, max_len)
        data = torch.LongTensor(data).transpose(0, 1)

        return data, np.array(lengths)



































