import math
import numpy as np
import copy
import random
import torch
import logging
import heapq
from util import *
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from torch import nn
from sklearn.cluster import KMeans, DBSCAN
from util import collate_fn
from util import load_pkl
from text2vec import Similarity 
from text2vec.algorithm.distance import cosine_distance
from itertools import groupby
from sklearn.decomposition import PCA
# from text2vec import models
from collections import defaultdict
from methods.LM import LMFcExtractor
from methods.PCNN import CNNFcExtractor
from methods.BiLSTM import BiLSTMFcExtractor
from gensim.models import Word2Vec
import gensim
logger = logging.getLogger(__name__)
class Select():
    def __init__(self,cfg,unlabeled_ds):
        super(Select, self).__init__()
        self.cfg = cfg
        self.unlabeled_ds =unlabeled_ds
        self.all_y_pred_probability = None 
        self.cur_labeled_ds = []
        self.select= None  
        self.size = 0

    def get_divided_by_select(self,cur_labeled_ds, unlabeled_ds,select):
        logger.info(f'select index:{select}')
        new_unlabeled_ds = []
        for index, sen in enumerate(unlabeled_ds):
            if index in select:
                cur_labeled_ds.append(sen)
            else:
                new_unlabeled_ds.append(sen)
        return cur_labeled_ds, new_unlabeled_ds

    def get_divided_by_select_2(self):
        if len(self.select) ==0:
            return None

        if len(self.select)>= self.size:
            select = self.select[:self.size]
            self.select = self.select[self.size:]
        else:
            select = self.select
            self.select =[]
        logger.info(f'select index:{select}')
        for index, sen in enumerate(self.unlabeled_ds):
            if index in select:
                self.cur_labeled_ds.append(sen)
        return self.cur_labeled_ds

    def get_probility(self, model,token,args):
        model.eval()
        all_y_pred = np.empty((0,args.max_len))
        tokenizer, id2 , l2id = token
        predicate2id,id2predicate = id2[0],id2[1]
        label2id,id2label = l2id[0],l2id[1]
        dataloader = data_generator(args,self.unlabeled_ds, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.batch_size,random=True,is_train = False)
        for i ,data_source in enumerate(dataloader):

            data_source = [torch.tensor(d).to("cuda") for d in data_source[:-1]]
            batch_token_ids, batch_mask = data_source 
            with torch.no_grad():
                y_pred, _ = model(token_ids=batch_token_ids,mask_token_ids = batch_mask, alpha=0, domain = False)
            conv_layer = torch.nn.Conv3d(args.max_len, args.max_len, kernel_size=(args.max_len,self.cfg.num_relations,3), stride=[1,1,1], padding=0, dilation=[1,1,1], groups=1, bias=True).cuda()
            y_pred1 = conv_layer(y_pred)
            y_pred2 = torch.squeeze(y_pred1,dim=-1)
            y_pred3 = torch.squeeze(y_pred2,dim=-1)
            y_pred = torch.squeeze(y_pred3,dim=-1)

            y_pred = y_pred.cpu().detach().numpy()
            all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0) 

        all_y_pred = torch.from_numpy(all_y_pred) 
        self.all_y_pred_probability = nn.functional.softmax(all_y_pred, dim=1).numpy() 
    def init_source_rate(self,size):
        if len(self.select) ==0:
            return None
        self.size = int(size)
        if len(self.select)>= self.size:
            select = self.select[:self.size] 
            self.select = self.select[self.size:]
        else:
            select = self.select
            self.select =[]
        logger.info(f'select index:{select}')
        s = copy.deepcopy(self.unlabeled_ds)
        for index, sen in enumerate(self.unlabeled_ds):
            if index in select:
                self.cur_labeled_ds.append(sen)
                s.remove(sen)
        self.unlabeled_ds = copy.deepcopy(s)
        return self.cur_labeled_ds,self.unlabeled_ds
    
    def uncertainty_sample(self):
        type1 = self.cfg.concrete
        if type1 == 'least_confident':
            tmp = self.all_y_pred_probability.max(axis=1)
            self.select = heapq.nsmallest(len(tmp), range(len(tmp)), tmp.take)
        elif type1 == 'margin_sampling':
            res = np.empty(0)
            ttmp = np.vsplit(self.all_y_pred_probability, self.all_y_pred_probability.shape[0])
            for tmp in ttmp:
                tmp = np.squeeze(tmp)
                first_two = tmp[np.argpartition(tmp, -2)[-2:]]
                res = np.concatenate((res, np.array([abs(first_two[0] - first_two[1])])), axis=0)
            self.select = heapq.nsmallest(len(res), range(len(res)), res.take) 
        elif type1 == 'entropy_sampling':
            res = np.empty(0)
            ttmp = np.vsplit(self.all_y_pred_probability, self.all_y_pred_probability.shape[0])
            for tmp in ttmp:
                tmp = np.squeeze(tmp) 
                res = np.concatenate((res, np.array([tmp.dot(np.log2(tmp))])), axis=0)
            
            self.select = heapq.nsmallest(len(res), range(len(res)), res.take)
        else:
            assert ('uncertainty concrete choose error')
        