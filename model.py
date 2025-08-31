import torch.nn as nn
from functions import ReverseLayerF
        
from transformers.modeling_bert import BertModel,BertPreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config) 
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,  
        encoder_hidden_states, 
        encoder_attention_mask 
    ):
        self_attention_outputs = self.attention(hidden_states)  
        attention_output = self_attention_outputs[0] 
        outputs = self_attention_outputs[1:]  

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :] 
        elif encoder_attention_mask.dim() == 2: #
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] 
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask)
        attention_output = cross_attention_outputs[0] 
        outputs = outputs + cross_attention_outputs[1:]  

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) 
        outputso = (layer_output,) + outputs 
        return outputso

class legalTALE(BertPreTrainedModel):
    def __init__(self, config):
        super(legalTALE, self).__init__(config)
        self.tabel = 500
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(768, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 72 * 72, self.tabel))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(self.tabel))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.tabel, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        self.bert=BertModel(config=config)

        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.Lr_e1=nn.Linear(config.hidden_size,config.hidden_size)
        self.Lr_e2=nn.Linear(config.hidden_size,config.hidden_size) 

        self.elu=nn.ELU()
        self.Cr = nn.Linear(config.hidden_size, config.num_p*config.num_label) 

        self.Lr_e1_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size) 
        self.Lr_e2_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size) 

        self.rounds=config.rounds

        self.e_layer=DecoderLayer(config)

        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1) 
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)

    def forward(self, token_ids, mask_token_ids,alpha,domain=True):
        h = token_ids
        embed=self.get_embed(token_ids, mask_token_ids)  
        L=embed.shape[1]
        e1 = self.Lr_e1(embed) 
        e2 = self.Lr_e2(embed) 

        for i in range(self.rounds):

            e11 = e1.unsqueeze(2).expand(-1, -1, L, -1).reshape(-1, L, L, 768)

            e22 = e2.unsqueeze(1).expand(-1,L,  -1, -1).reshape(-1, L, L, 768)

            e31 = self.block_hadamard_product(A=e11,B=e22,P=10)

            h = self.elu(e31)  
            
            B, L = h.shape[0], h.shape[1] 

            table_logist = self.Cr(h)  

            if i!=self.rounds-1:
                                                           
                table_e1 = table_logist.max(dim=2).values 
                table_e2 = table_logist.max(dim=1).values 
                e1_ = self.Lr_e1_rev(table_e1) 
                e2_ = self.Lr_e2_rev(table_e2) 

                e1=e1+self.e_layer(e1_,embed,mask_token_ids)[0]
                e2=e2+self.e_layer(e2_,embed,mask_token_ids)[0]

        domain_output = None
        if domain== True:
            h = h.transpose(1,3)
            feature = self.feature(h)
            l = feature.size()[-1]
            feature = feature.view(-1, 50 * l * l)  
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label]), domain_output

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())  #id转为向量表示  101 -> 101*768
        embed=bert_out[0]
        embed=self.dropout(embed)
        return embed
    def block_hadamard_product(self, A, B, P):
        assert A.shape == B.shape, "The two input tensors should have the same shape"
        a, b, c, d = A.shape
        assert b % P == 0, "The block size should evenly divide the second dimension of the tensor"

        block_size = b // P
        C_blocks = []
        for i in range(P):
            start_idx = i * block_size
            end_idx = (i+1) * block_size
            A_block = A[:, start_idx:end_idx, :, :]
            B_block = B[:, start_idx:end_idx, :, :]
            C_block = torch.mul(A_block, B_block)
            C_blocks.append(C_block)

        C = torch.cat(C_blocks, dim=1)
        return C