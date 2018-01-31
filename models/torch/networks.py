import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_p=0.2):
        super(EncoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.dropout1(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.dropout2(output)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, dropout_p=0.2):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.embed_size, self.hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout1(output)
        output, hidden = self.gru(output, hidden)
        output = self.dropout2(output)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, dropout_p=0.2):
        super(AttnDecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.attn_combine = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout1(embedded)

        attn_energies = self.attn(torch.cat((hidden[0].repeat(len(encoder_outputs), 1), encoder_outputs), 1))
        attn_energies = attn_energies.transpose(0, 1)
        attn_weights = F.softmax(attn_energies, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                        encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.dropout2(output).unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        output = self.dropout3(output)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class NewAttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, max_length, dropout_p=0.2):
        super(NewAttnDecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.attn = nn.Linear(self.embed_size + self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout1(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.dropout2(output).unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        output = self.dropout3(output)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class RNNLangmod(nn.Module):
    def __init__(self, embed_layer, rnn_layer, hidden_size, output_size, dropout_p=0.5, output_layer=None):
        super(RNNLangmod, self).__init__()
        self.embed_size = embed_layer.embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = embed_layer
        self.rnn = rnn_layer
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(self.embed_size, self.embed_size + self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = output_layer if output_layer is not None else nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout1(embedded)

        output = self.fc1(embedded)
        output, hidden = self.rnn(output, hidden)
        output = self.dropout2(output)

        output = F.relu(self.fc2(output[0]))

        output = F.log_softmax(self.output_layer(output), dim=1)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result