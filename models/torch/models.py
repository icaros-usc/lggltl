import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class TestEncoderRNN(nn.Module):
#     def __init__(self, input_size, embed_size, hidden_size, weights_matrix, use_cuda):
#         super(TestEncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.init_weights = weights_matrix
#         self.use_cuda = use_cuda
#         self.embedding, num_embeddings, self.embed_size = self.create_emb_layer(weights_matrix, False)
#         self.gru = nn.GRU(self.embed_size, self.hidden_size)
#
#     def create_emb_layer(self, weights_matrix, non_trainable=False):
#         num_embeddings, embedding_dim = weights_matrix.shape
#         emb_layer = nn.Embedding(num_embeddings, embedding_dim)
#         emb_layer.load_state_dict({'weight': torch.FloatTensor(weights_matrix)})
#         if non_trainable:
#             emb_layer.weight.requires_grad = False
#         return emb_layer, num_embeddings, embedding_dim
#
#     def forward(self, input, hidden):
#         return self.gru(self.embedding(input).view(1, 1, -1), hidden)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, use_cuda, dropout_p=0.2):
        super(EncoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.dropout(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.dropout(output)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, use_cuda, dropout_p=0.2):
        super(AttnDecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.attn_combine = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_energies = self.attn(torch.cat((hidden[0].repeat(len(encoder_outputs), 1), encoder_outputs), 1))
        attn_energies = attn_energies.transpose(0, 1)
        attn_weights = F.softmax(attn_energies, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.dropout(output).unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        output = self.dropout(output)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def inherit(self, langmod):
        self.embedding = langmod.embed
        self.gru = langmod.rnn
        self.out = langmod.decoder
