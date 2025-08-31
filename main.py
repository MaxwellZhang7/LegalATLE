import random
import fitlog
import hydra
from hydra import utils
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer
from model import legalTALE
from util import *
from tqdm import tqdm
import torch.nn as nn
import torch
from transformers.modeling_bert import BertConfig
import json
import argparse
from data_select import Select
import logging
import sys
import itertools
from test_t import evaluate
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a+')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

@hydra.main(config_path='config',config_name='config')
def main(cfg):
    cwd = utils.get_original_cwd() 
    cfg.cwd = cwd
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--dataset', default='tem', type=str) 
    parser.add_argument('--rounds', default=5, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-5 ,type=float)
    parser.add_argument('--num_train_epochs', default=50, type=int)
    parser.add_argument('--max_len', default=300, type=int)
    parser.add_argument('--select_method', default="uncertainty", type=str)#uncertainty
    parser.add_argument('--primary_data_rate', default=0.1, type=float)#
    # parser.add_argument('--full_data_rate', default=1, type=float)
    parser.add_argument('--select_data_rate', default=0.07, type=float)#

    parser.add_argument('--train', default="train", type=str)
    parser.add_argument('--file_id', default="99", type=str)
    parser.add_argument('--fix_bert_embeddings', default=True, type=bool)
    parser.add_argument('--bert_vocab_path', default="legalTALE/PLM/bert-cn-wwm/vocab.txt", type=str)
    parser.add_argument('--bert_config_path', default="legalTALE/PLM/bert-cn-wwm/config.json", type=str)
    parser.add_argument('--bert_model_path', default="legalTALE/PLM/bert-cn-wwm/pytorch_model.bin", type=str)
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--min_num', default=1e-7, type=float)
    parser.add_argument('--base_path', default="legalTALE/dataset", type=str)

    args = parser.parse_args()

    source_dataset_name = 'drug'
    target_dataset_name = 'theft'
    model_root = 'legalTALE/models'
    cuda = True
    cudnn.benchmark = True
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(1000)
    if torch.cuda.is_available(): 
        device = torch.device('cuda',args.cuda_id)
    else:
        device = torch.device("cpu")

    label_list=["N/A","MMH","MMT"]
    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i

    # load data
    scource_path=os.path.join(args.base_path,args.dataset,"drug.json")
    target_path=os.path.join(args.base_path,args.dataset,"theft.json")
    rel2id_path=os.path.join(cfg.cwd,args.base_path,args.dataset,"rel2id.json")
    scource_path_test=os.path.join(args.base_path,args.dataset,"drug_test.json")
    target_path_test=os.path.join(args.base_path,args.dataset,"theft_test.json")
    output_path=os.path.join(args.base_path,args.dataset,"output",args.file_id)
    test_pred_path = os.path.join(output_path, "test_pred.json")
    scource_data_test = json.load(open(scource_path_test))
    target_data_test = json.load(open(target_path_test,encoding= "gb18030"))

    id2predicate, predicate2id = json.load(open(rel2id_path)) 
    scource_data = json.load(open(scource_path))
    target_data = json.load(open(target_path,encoding= "gb18030"))   
    all_size = len(target_data)
    full_size = all_size

    tokenizer = Tokenizer(args.bert_vocab_path)

    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p=len(id2predicate)
    config.num_label=len(label_list)
    config.rounds=args.rounds
    config.fix_bert_embeddings=args.fix_bert_embeddings

    scource_dataloader_test = data_generator(args,scource_data_test, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)
    
    target_dataloader_test = data_generator(args,target_data_test, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)
    target_dataloader=data_generator(args,target_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],1,random=True,is_train = True)
    len_target = len(target_dataloader)

    # load model
    my_net = legalTALE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    my_net = my_net.cuda()

    # setup optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in my_net.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in my_net.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
        "weight_decay": 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=args.min_num)
    loss_domain = torch.nn.NLLLoss()
    loss_domain = loss_domain.cuda()
    tri = nn.TripletMarginWithDistanceLoss(reduction="none")
    crossentropy=nn.CrossEntropyLoss(reduction="none")

    # training
    best_f1_t = 0.0
    epoch_num =0

    t_total = args.num_train_epochs*len_target
    for epoch in range(epoch_num):

        data_target_iter = iter(target_dataloader)
        for i in range(len_target):
            for data_target in data_target_iter:
                data_target = [torch.tensor(d).to("cuda") for d in data_target[:-1]]
                batch_token_ids, batch_mask,batch_label,batch_mask_label = data_target
                batch_size = len(batch_token_ids)

                class_output_t, domain_output = my_net(token_ids=batch_token_ids,mask_token_ids = batch_mask, alpha=0, domain = False)
                table = class_output_t
                table=table.reshape([-1,len(label_list)]) 
                batch_label=batch_label.reshape([-1]) 
                sloss = tri(table,table,table)
                loss=crossentropy(table,batch_label.long()) 
                loss *= sloss
                err_t_label=(loss*batch_mask_label.reshape([-1])).sum()
                err_t_label.backward(retain_graph=True) # 

                torch.nn.utils.clip_grad_norm_(my_net.parameters(), args.max_grad_norm)
                optimizer.step() 
                # scheduler.step()
                my_net.zero_grad()
                if i % 20 ==0:
                    sys.stdout.write('\r select epoch: %d, [iter: %d / all %d steps], err_t_label: %f' \
                        % (epoch+1, i + 1, len_target, err_t_label.data.cpu().numpy()))
                    sys.stdout.flush()  
                break

        f1_t, precision_t, recall_t = evaluate(args, tokenizer, id2predicate, id2label, label2id, my_net, target_dataloader_test,test_pred_path,alpha=0,target=False)#domain
        sys.stdout.write('\n select source data: %s-> f1_t: %f, precision_t: %f,recall_t: %f \n' % (target_dataset_name,f1_t, precision_t, recall_t))
    

    s = Select(cfg, scource_data) 
    l1 = len_target

    
    selected_size = int(l1*args.primary_data_rate)
    print(str(selected_size))
    l2 =args.select_data_rate*l1

    if l2 >0 and l2<1:
        selected_size_pre = 1
    else:
        selected_size_pre = int(l2)

    print(str(selected_size_pre))
    token = tokenizer,[predicate2id,id2predicate],[label2id,id2label]
    if args.select_method == "uncertainty":
            s.get_probility(my_net,token,args)
            s.uncertainty_sample()
            scource_data_current, scource_data_unlabeled= s.init_source_rate(selected_size) 

    
    print(str(len(scource_data_current))+": "+str(len(scource_data_unlabeled)))
    
    # del my_net,
    torch.cuda.empty_cache()

    t_total = len(scource_data_current) * args.num_train_epochs 
    weights = np.random.rand(4)
    weight = weights/np.sum(weights)
    weight1 = nn.Parameter(torch.tensor([weight[0]],requires_grad=True,device=device))
    weight2 = nn.Parameter(torch.tensor([weight[1]],requires_grad=True,device=device))
    weight3 = nn.Parameter(torch.tensor([weight[2]],requires_grad=True,device=device))
    weight4 = nn.Parameter(torch.tensor([weight[3]],requires_grad=True,device=device))
    weight_optimizer = torch.optim.Adam([weight1,weight2,weight3,weight4],lr = 1e-3)
    my_net1 = my_net
    
    for epoch in range(args.num_train_epochs):
        if full_size <= len(scource_data_current):
            # del s
            pass
        t_total = len(scource_data_current) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total)
        weight_scheduler = get_linear_schedule_with_warmup(weight_optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total)
        scource_dataloader = data_generator(args,scource_data_current, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.batch_size,random=True,is_train = True)
        len_dataloader_min = min(len(scource_data_current), len_target)
        data_source_iter = iter(scource_dataloader)
        data_target_iter = iter(target_dataloader)
        torch.cuda.empty_cache()
        L_BNM = -torch.norm(,'nuc')
        for i in range(len_dataloader_min): 
            p = float(i + epoch * len_dataloader_min) / args.num_train_epochs / len_dataloader_min
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            for index , [data_source,data_target] in enumerate(zip(data_source_iter,data_target_iter)):
                data_source = [torch.tensor(d).to("cuda") for d in data_source[:-1]]
                batch_token_ids, batch_mask,batch_label,batch_mask_label = data_source
                batch_size = len(batch_token_ids)
                domain_label = torch.zeros(batch_size).long()
                if cuda:
                    domain_label = domain_label.cuda()
                class_output_s, domain_output = my_net1(token_ids=batch_token_ids,mask_token_ids = batch_mask, alpha=alpha, domain = True)
                table = class_output_s
                table=table.reshape([-1,len(label_list)]) 
                batch_label=batch_label.reshape([-1]) 
                sloss = tri(table,table,table)
                loss=crossentropy(table,batch_label.long()) 
                loss *= sloss
                err_s_label=(loss*batch_mask_label.reshape([-1])).sum()
                err_s_domain = loss_domain(domain_output, domain_label)
                torch.cuda.empty_cache()
                data_target = [torch.tensor(d).to("cuda") for d in data_target[:-1]]
                batch_token_ids_t, batch_mask_t,batch_label_t,batch_mask_label_t = data_target
                batch_size = len(batch_token_ids_t)
                domain_label = torch.ones(batch_size).long()
                if cuda:
                    domain_label = domain_label.cuda()
                class_output_t, domain_output = my_net1(token_ids=batch_token_ids_t,mask_token_ids = batch_mask_t, alpha=alpha, domain= True)
                err_t_domain = loss_domain(domain_output, domain_label)
                #
                table = class_output_t
                table=table.reshape([-1,len(label_list)]) 
                batch_label_t=batch_label_t.reshape([-1]) 
                tloss = tri(table,table,table)
                loss=crossentropy(table,batch_label_t.long()) 
                loss *= tloss
                err_t_label=(loss*batch_mask_label_t.reshape([-1])).sum() 
                torch.cuda.empty_cache()
                break
            torch.cuda.empty_cache()
            err =  weight1*err_s_label +weight2*err_t_domain + weight3*err_s_domain+weight4*err_t_label + 
            err.backward(retain_graph=True) 
            torch.nn.utils.clip_grad_norm_(my_net1.parameters(), args.max_grad_norm) 
            optimizer.step()
            scheduler.step()
            weight_optimizer.step()
            weight_scheduler.step()
            my_net1.zero_grad()
            if i % 20 ==0:
                sys.stdout.write('\r epoch: %d, [iter: %d / all %d steps], err_s_label: %f, err_t_label: %f,err_s_domain: %f, err_t_domain: %f' \
                    % (epoch+1, i + 1, len_dataloader_min, err_s_label.data.cpu().numpy(),err_t_label.data.cpu().numpy(),
                        err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
                sys.stdout.flush()
        torch.cuda.empty_cache()
        time.sleep(0.003)
        if epoch+1 >= 2:
            alpha = 0
            print("\n"+str(epoch)+"epochs...")
            f1_s, precision_s, recall_s = evaluate(args, tokenizer, id2predicate, id2label, label2id, my_net1, scource_dataloader_test,test_pred_path,alpha,target=False)
            sys.stdout.write('\n dataset: %s-> f1_s: %f, precision_s: %f,recall_s: %f .' % (source_dataset_name,f1_s, precision_s, recall_s))
            
            f1_t, precision_t, recall_t = evaluate(args, tokenizer, id2predicate, id2label, label2id, my_net1, target_dataloader_test,test_pred_path,alpha,target=False)
            sys.stdout.write('\n dataset: %s-> f1_t: %f, precision_t: %f,recall_t: %f \n' % (target_dataset_name,f1_t, precision_t, recall_t))
            if f1_t > best_f1_t:
                # best_f1_s = f1_s
                best_f1_t = f1_t
                torch.save(my_net1, '{0}/drug_theft_model_epoch_best.pth'.format(model_root))
            
        if len(scource_data_current) < full_size and epoch% 3 == 0:
            print(str(len(scource_data_current))+":"+str(all_size))
            if args.select_method =="uncertainty":
                s.get_probility(my_net1, token,args)
                s.uncertainty_sample()
                scource_data_current, scource_data_unlabeled= s.init_source_rate(selected_size_pre)
            
            print(str(len(scource_data_current))+": "+str(len(scource_data_unlabeled)))
        torch.cuda.empty_cache()
    print('============ Summary ============= \n')
    print('f1 of the %s dataset: %f' % (target_dataset_name, best_f1_t))
    print('Corresponding model was save in ' + model_root + '/drug_theft_model_epoch_best.pth')

if __name__ == "__main__":
    import time
    cur = time.time
    sys.stdout = Logger('a.log', sys.stdout)
    sys.stderr = Logger('a.log_file', sys.stderr)
    main()