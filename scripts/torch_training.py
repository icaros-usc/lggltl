"""Using this script to train the language model"""

import itertools
import os.path as osp
import random
import time

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

from baselines.lggltl.models.torch.models import EncoderRNN, AttnDecoderRNN
from baselines.lggltl.scripts.lang_utils import prepare_data, vectors_for_input_lang
from baselines.lggltl.models.torch.utils import timeSince
from baselines.lggltl.scripts.train_eval_utils import *
from baselines.lggltl.scripts.torch_evaluating import evaluate_randomly

use_cuda = torch.cuda.is_available()  # whether use cuda
# set the random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed) if not use_cuda else torch.cuda.manual_seed(seed)

data_path = osp.join(osp.dirname(osp.dirname(__file__)), 'data')
src_path, tar_path = osp.join(data_path, 'hard_pc_src_syn.txt'), osp.join(data_path, 'hard_pc_tar_syn.txt')

input_lang, output_lang, pairs, max_src_len, max_tar_len = prepare_data(src_path, tar_path, False)
random.shuffle(pairs)

embed_size = 50
hidden_size = 256

# GLOVE = True  # whether use global vectors for initializing word representation
# if GLOVE:
#     glove_path = osp.join(osp.dirname(osp.dirname(__file__)), 'glove')
#     glove_map = vectors_for_input_lang(input_lang, glove_path)
#     glove_encoder = TestEncoderRNN(input_lang.n_words, embed_size, hidden_size, glove_map)

encoder1 = EncoderRNN(input_lang.n_words, embed_size, hidden_size, use_cuda)
attn_decoder1 = AttnDecoderRNN(embed_size, hidden_size, output_lang.n_words, use_cuda)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()


def train(input_variable, target_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_src_length,
          use_cuda, teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_src_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(input_lang, output_lang, encoder, decoder, samples, n_iters,
                max_src_len, use_cuda, print_every=1000, plot_every=100, lr=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    training_pairs = itertools.cycle(iter([variables_from_pair(input_lang, output_lang, s, use_cuda) for s in samples]))
    criterion = nn.NLLLoss()

    for i in range(1, n_iters + 1):
        training_pair = next(training_pairs)
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_src_len, use_cuda)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters), i, i / n_iters * 100, print_loss_avg))


MODE = 0

if MODE == 0:
    train_iters(input_lang, output_lang, encoder1, attn_decoder1, pairs, 10000, max_src_len, use_cuda)
    encoder1.eval()
    attn_decoder1.eval()
    evaluate_randomly(input_lang, output_lang, encoder1, attn_decoder1, pairs, max_src_len, use_cuda)
