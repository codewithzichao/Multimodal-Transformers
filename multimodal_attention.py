import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy
import math

def SelfAttention(Q, K, V, mask=None, dropout=None):
    d_k = Q.shape[-1]

    logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, value=torch.tensor(-1e9))

    attn_weights = F.softmax(logits, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    attn_values = torch.matmul(attn_weights, V)

    return attn_values, attn_weights


class MHAttention(nn.Module):
    def __init__(self, head_num, d_model, dropout_rate=0.1):
        super(MHAttention, self).__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_model // self.head_num
        self.dropout_rate = dropout_rate

        assert self.d_model % self.head_num == 0

        self.linears = nn.ModuleList( \
            [deepcopy(nn.Linear(in_features=self.d_model, out_features=self.d_model)) for i in range(4)])

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, Q, K, V, mask=None):
        # mask是一个三维的tensor
        if mask is not None:
            mask = mask.unsqueeze(dim=1)
        batch_size = Q.shape[0]

        Q, K, V = [l(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2) for l, x in \
                   zip(self.linears, [Q, K, V])]

        attn_values, attn_weights = SelfAttention(Q, K, V, mask, dropout=self.dropout)
        attn_values = attn_values.transpose(1, 2).contiguous()
        attn_values = attn_values.view(batch_size, -1, self.d_k * self.head_num)

        return self.linears[-1](attn_values)

class LayerNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2=nn.Parameter(torch.ones(hidden_size))
        self.b_2=nn.Parameter(torch.ones(hidden_size))

        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)

        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class FFN(nn.Module):
    def __init__(self,hidden_size,hidden_ffn_size,dropout_rate):
        super(FFN, self).__init__()

        self.fc1=nn.Linear(in_features=hidden_size,out_features=hidden_ffn_size)
        self.fc2=nn.Linear(in_features=hidden_ffn_size,out_features=hidden_size)
        self.dropout=nn.Dropout(p=dropout_rate)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self,hidden_size,head_num,dropout_rate,hidden_ffn_size):
        super(TransformerLayer, self).__init__()

        self.mha_ttn=MHAttention(head_num=head_num,d_model=hidden_size,dropout_rate=dropout_rate)
        self.ln=LayerNorm(hidden_size=hidden_size)
        self.ffn=FFN(hidden_size=hidden_size,hidden_ffn_size=hidden_ffn_size,dropout_rate=dropout_rate)

    def forward(self,x,y,z,mask=None):

        sub_layer1_result=self.mha_ttn(x,y,z,mask)
        x=self.ln(x+sub_layer1_result)

        sub_layer2_result=self.ffn(x)
        x=self.ln(x+sub_layer2_result)

        return x


class MultimodalAttention(nn.Module):
    def __init__(self,hidden_size,head_num,dropout_rate,hidden_ffn_size):
        super(MultimodalAttention, self).__init__()

        self.hidden_size=hidden_size
        self.image2text_layer=TransformerLayer(self.hidden_size,head_num,dropout_rate,hidden_ffn_size)
        self.text2image_layer=TransformerLayer(self.hidden_size,head_num,dropout_rate,hidden_ffn_size)

        self.W_image=nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size,bias=False)
        self.W_text=nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size,bias=False)


    def forward(self,image_rep,text_rep,mask=None):
        P=self.image2text_layer(image_rep,text_rep,text_rep,mask)
        A=self.image2text_layer(text_rep,P,P,mask)

        B=self.text2image_layer(text_rep,image_rep,image_rep,mask)


        return A,B







