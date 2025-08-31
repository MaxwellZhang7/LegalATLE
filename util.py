#! -*- coding:utf-8 -*-
import numpy as np
import random
from copy import deepcopy
import os
import pickle
import torch
import torch
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)
def load_pkl(fp, verbose=True):
    if verbose:
        logger.info(f'load data from {fp}')
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data
def collate_fn(cfg):
    def collate_fn_intra(batch):
        batch.sort(key=lambda data: data['seq_len'], reverse=True)

        max_len = batch[0]['seq_len']

        def _padding(x, max_len):
            return x + [0] * (max_len - len(x))

        x, y = dict(), []
        word, word_len = [], []
        head_pos, tail_pos = [], []
        pcnn_mask = []
        for data in batch:
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(data['seq_len'])

            y.append(int(data['rel2idx']))

            if cfg.model.model_name != 'lm':
                head_pos.append(_padding(data['head_pos'], max_len))
                tail_pos.append(_padding(data['tail_pos'], max_len))
                if cfg.model.model_name == 'cnn':
                    if cfg.model.use_pcnn:
                        pcnn_mask.append(_padding(data['entities_pos'], max_len))
        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        y = torch.tensor(y)
        if cfg.model.model_name != 'lm':
            x['head_pos'] = torch.tensor(head_pos)
            x['tail_pos'] = torch.tensor(tail_pos)
            if cfg.model.model_name == 'cnn' and cfg.model.use_pcnn:
                x['pcnn_mask'] = torch.tensor(pcnn_mask)
            if cfg.model.model_name == 'gcn':
                B, L = len(batch), max_len
                adj = torch.empty(B, L, L).random_(2)
                x['adj'] = adj
        return x, y

    return collate_fn_intra


class CustomDataset(Dataset):

    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)
def print_config(args):
    config_path=os.path.join(args.base_path, args.dataset, "output", args.file_id,"config.txt")
    with open(config_path,"w",encoding="utf-8") as f:
        for k,v in sorted(vars(args).items()):
            print(k,'=',v,file=f)

def set_seed():

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def mat_padding(inputs, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[0] for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[0])
        pad_width[1] = (0, length - x.shape[0])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def tuple_mat_padding(inputs,dim=1, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[1] = (0, length - x.shape[dim])
        pad_width[2] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def sequence_padding(inputs,dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs]) 
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding) 
        outputs.append(x)
    return np.array(outputs)


def judge(ex):
    for s,p,o in ex["triple_list"]:
        if s=='' or o=='' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True


class DataGenerator(object):
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        for d in self.__iter__(True):
            yield d


class Vocab(object):
    def __init__(self, filename, load=False, word_counter=None, threshold=0):
        if load:
            assert os.path.exists(filename), "Vocab file does not exist at " + filename
            # load from file and ignore all other params
            self.id2word, self.word2id = self.load(filename)
            self.size = len(self.id2word)
            print("Vocab size {} loaded from file".format(self.size))
        else:
            print("Creating vocab from scratch...")
            assert word_counter is not None, "word_counter is not provided for vocab creation."
            self.word_counter = word_counter
            if threshold > 1:
                # remove words that occur less than thres
                self.word_counter = dict([(k, v) for k, v in self.word_counter.items() if v >= threshold])
            self.id2word = sorted(self.word_counter, key=lambda k: self.word_counter[k], reverse=True)
            # add special tokens to the beginning
            self.id2word = ['**PAD**', '**UNK**'] + self.id2word
            self.word2id = dict([(self.id2word[idx], idx) for idx in range(len(self.id2word))])
            self.size = len(self.id2word)
            self.save(filename)
            print("Vocab size {} saved to file {}".format(self.size, filename))

    def load(self, filename):
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])
        return id2word, word2id

    def save(self, filename):
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(self.id2word, outfile)
        return

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.word2id[w] if w in self.word2id else constant.VOCAB_UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]

    def get_embeddings(self, word_vectors=None, dim=100):
        self.embeddings = np.zeros((self.size, dim))
        if word_vectors is not None:
            assert len(list(word_vectors.values())[0]) == dim, \
                "Word vectors does not have required dimension {}.".format(dim)
            for w, idx in self.word2id.items():
                if w in word_vectors:
                    self.embeddings[idx] = np.asarray(word_vectors[w])
        return self.embeddings
def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class data_generator(DataGenerator):
    def __init__(self,args,train_data, tokenizer,predicate_map,label_map,batch_size,random=False,is_train=True):
        super(data_generator,self).__init__(train_data,batch_size)
        self.max_len=args.max_len
        self.tokenizer=tokenizer
        self.predicate2id,self.id2predicate=predicate_map
        self.label2id,self.id2label=label_map
        self.random=random
        self.is_train=is_train

    def __iter__(self):
        batch_token_ids, batch_mask = [], []
        batch_label=[]
        batch_mask_label=[]
        batch_ex=[]
        for is_end, d in self.sample(self.random):
            if judge(d)==False: 
                continue

            token_ids,mask = self.tokenizer.encode(
               d['text'],max_length=self.max_len,first_length=self.max_len,)
            if self.is_train:
                spoes = {}  
                for s, p, o in d['triple_list']:
                    s = self.tokenizer.encode(s)[0][1:-1]
                    p = self.predicate2id[p]
                    o = self.tokenizer.encode(o)[0][1:-1]
                    s_idx = search(s, token_ids)
                    o_idx = search(o, token_ids) 
                    if s_idx != -1 and o_idx != -1:
                        s = (s_idx, s_idx + len(s) - 1) 
                        o = (o_idx, o_idx + len(o) - 1, p) 
                        if s not in spoes: 
                            spoes[s] = []
                        spoes[s].append(o)
                if spoes:
                    label=np.zeros([len(token_ids), len(token_ids),len(self.id2predicate)]) 
                    for s in spoes:
                        s1,s2=s
                        for o1,o2,p in spoes[s]:
                            if s1!=s2 and o1!=o2:
                                label[s1, o1,p] = self.label2id["MMH"]
                                label[s2, o2,p] = self.label2id["MMT"]

                    mask_label=np.ones(label.shape)
                    mask_label[0,:,:]=0 
                    mask_label[-1,:,:]=0
                    mask_label[:,0,:]=0
                    mask_label[:,-1,:]=0

                    for a,b in zip([batch_token_ids, batch_mask,batch_label,batch_mask_label,batch_ex],
                                   [token_ids,mask,label,mask_label,d]):
                        a.append(b)

                    if len(batch_token_ids) == self.batch_size or is_end:
                        batch_token_ids, batch_mask=[sequence_padding(i) for i in [batch_token_ids, batch_mask]] 
                        batch_label=mat_padding(batch_label)
                        batch_mask_label=mat_padding(batch_mask_label)
                        yield [
                            batch_token_ids, batch_mask,
                            batch_label,
                            batch_mask_label,batch_ex
                        ]
                        batch_token_ids, batch_mask = [], []
                        batch_label=[]
                        batch_mask_label=[]
                        batch_ex=[]

            else:
                for a, b in zip([batch_token_ids, batch_mask, batch_ex],
                                [token_ids, mask, d]):
                    a.append(b)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                    yield [
                        batch_token_ids, batch_mask, batch_ex
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_ex = []
    def __next__():
        pass
def seq_len_to_mask(seq_len, max_len=None, mask_pos_to_true=True):
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask