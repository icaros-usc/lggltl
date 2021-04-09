import torch
from torch.autograd import Variable


def variables_from_pair(input_lang, output_lang, pair, use_cuda):
    input_variable = variable_from_sentence(input_lang, ' '.join(list(reversed(pair[0].split()))), use_cuda)
    target_variable = variable_from_sentence(output_lang, pair[1], use_cuda)
    return input_variable, target_variable


SOS_token = 0
EOS_token = 1
UNK_token = 2


def variable_from_sentence(lang, sentence, use_cuda):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
