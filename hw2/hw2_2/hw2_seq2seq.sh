#!/bin/bash
wget 'https://www.dropbox.com/s/6yubcujofybgnna/encoder.pt?dl=1' -O encoder.pt
wget 'https://www.dropbox.com/s/61iyscv048nwr3s/decoder.pt?dl=1' -O decoder.pt
wget 'https://www.dropbox.com/s/lkshbwo1t1rq227/idx2word.pk?dl=1' -O idx2word.pk
wget 'https://www.dropbox.com/s/8vgcmy1tzm5kt1w/word2idx.pk?dl=1' -O word2idx.pk
python test.py $1 $2