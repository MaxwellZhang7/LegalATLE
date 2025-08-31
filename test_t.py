import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
from tqdm import tqdm
import json
from bert4keras.tokenizers import Tokenizer
from transformers.modeling_bert import BertConfig
from util import *

def extract_spoes(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex, batch_token_ids, batch_mask,alpha,target):
    def get_pred_id(table,all_tokens):
        B, L, _, R, _ = table.shape 
        res = []
        for i in range(B):
            res.append([])
        table = table.argmax(axis=-1)  #
        all_loc = np.where(table != label2id["N/A"]) 
        res_dict = []
        for i in range(B):
            res_dict.append([])
        for i in range(len(all_loc[0])):
            token_n=len(all_tokens[all_loc[0][i]])
            if token_n-1 <= all_loc[1][i] \
                    or token_n-1<=all_loc[2][i] \
                    or 0 in [all_loc[1][i],all_loc[2][i]]:  
                continue
            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])
        for i in range(B):
            for l1, l2, r in res_dict[i]:

                if table[i, l1, l2, r] == label2id["MMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MMT"] and l1_ > l1 and l2_ > l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
        return res
    model.eval()
    with torch.no_grad():
        table,_=model(batch_token_ids, batch_mask,alpha,target) 
        table = table.cpu().detach().numpy() 
    all_tokens=[]
    for ex in batch_ex:
        tokens = tokenizer.tokenize(ex["text"], max_length=args.max_len)
        all_tokens.append(tokens)

    res_id=get_pred_id(table,all_tokens) 
    batch_spo=[[] for _ in range(len(batch_ex))]
    for b,ex in enumerate(batch_ex):
        text=ex["text"]
        tokens = all_tokens[b]
        mapping = tokenizer.rematch(text, tokens)
        for sh, st, r, oh, ot in res_id[b]:

            s=(mapping[sh][0], mapping[st][-1])
            o=(mapping[oh][0], mapping[ot][-1])

            batch_spo[b].append(
                (text[s[0]:s[1] + 1], id2predicate[str(r)], text[o[0]:o[1] + 1]))
    return batch_spo
def evaluate(args,tokenizer,id2predicate,id2label,label2id,model,dataloader,evl_path,alpha,target):

    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm()
    index = 0
    for batch in dataloader:
        index += 1
        batch_ex=batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch
        batch_spo=extract_spoes(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex,batch_token_ids, batch_mask,alpha,target)
        for i,ex in enumerate(batch_ex):
            R = set(batch_spo[i])
            T = set([(item[0], item[1], item[2]) for item in ex['triple_list']])
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            s = json.dumps({
                'text': ex['text'],
                'triple_list': list(T),
                'triple_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False)
            f.write(s + '\n')
    pbar.close()
    f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

