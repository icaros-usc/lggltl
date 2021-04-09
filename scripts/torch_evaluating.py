import random

from baselines.lggltl.scripts.train_eval_utils import *


def evaluate_randomly(input_lang, output_lang, encoder, decoder, pairs, max_length, use_cuda, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length, use_cuda)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_src_length, use_cuda):
    input_variable = variable_from_sentence(input_lang, ' '.join(list(reversed(sentence.split()))), use_cuda)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(torch.zeros(max_src_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_src_length, max_src_length)

    for di in range(max_src_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        if decoder_attention is not None:
            decoder_attentions[di] = decoder_attention.data

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]
