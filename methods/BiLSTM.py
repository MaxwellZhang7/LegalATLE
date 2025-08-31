import torch.nn as nn
from . import BasicModule
from . import Embedding, RNN


class BiLSTM(BasicModule.BasicModule):
    def __init__(self, cfg):
        super(BiLSTM, self).__init__()

        if cfg.dim_strategy == 'cat':
            cfg.model.input_size = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.model.input_size = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.bilstm = RNN(cfg)
        self.fc = nn.Linear(cfg.model.hidden_size, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        inputs = self.embedding(word, head_pos, tail_pos)
        out, out_pool = self.bilstm(inputs, lens)
        output = self.fc(out_pool)

        return output

class BiLSTMFcExtractor(nn.Module):
    def __init__(self, submodule):
        super(BiLSTMFcExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        inputs = self.submodule.embedding(word, head_pos, tail_pos)
        out, out_pool = self.submodule.bilstm(inputs, lens)
        return out_pool
