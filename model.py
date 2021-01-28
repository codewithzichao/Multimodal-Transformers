import torch
import torch.nn as nn
import torch.nn.functional as F
from multimodal_attention import MultimodalAttention
from torchvision.models import resnet101, resnet152
from transformers import ElectraModel, BertModel, XLMRobertaModel


class MyModel(nn.Module):
    def __init__(self, model_name, resnet_path, bert_path, dropout_rate, num_class):
        super(MyModel, self).__init__()

        self.model_name = model_name
        self.resnet_path = resnet_path
        self.bert_path = bert_path
        self.dropout_rate = dropout_rate
        self.num_class = num_class

        resnet = resnet152(pretrained=False)
        resnet.load_state_dict(torch.load(self.resnet_path), strict=False)
        for p in resnet.parameters():
            p.requires_grad = False

        resnet_fea = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*resnet_fea)

        if self.model_name == "electra":
            self.bert = ElectraModel.from_pretrained(self.bert_path)
        elif self.model_name == "xlm-bert":
            self.bert = BertModel.from_pretrained(self.bert_path)
        elif self.model_name == "xlm-roberta":
            self.bert = XLMRobertaModel.from_pretrained(self.bert_path)
        else:
            raise NotImplementedError

        # 512*4
        self.image_fc = nn.Linear(in_features=2048, out_features=self.bert.config.hidden_size)

        self.multimodal_attention_layer = MultimodalAttention(hidden_size=self.bert.config.hidden_size, head_num=12, dropout_rate=self.dropout_rate, \
                              hidden_ffn_size=self.bert.config.intermediate_size)

        self.temp_fc=nn.Linear(in_features=49,out_features=256)

        self.avg_pool_layer = nn.AdaptiveAvgPool1d(output_size=1)
        self.trans_fc = nn.Linear(in_features=self.bert.config.hidden_size * 2,\
                                  out_features=self.bert.config.hidden_size)
        self.fc = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.num_class)

    def forward(self, image, input_ids, attention_mask):

        image_rep = self.resnet(image)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        batch_size = image_rep.shape[0]
        channels = image_rep.shape[1]
        image_rep = image_rep.view(batch_size, -1, channels)
        image_rep = self.image_fc(image_rep)

        text_rep = bert_output[0]

        A, B = self.multimodal_attention_layer(image_rep, text_rep)

        x = torch.cat([A, B], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.avg_pool_layer(x).squeeze(dim=-1)

        x = self.trans_fc(x)
        x = self.fc(x)

        return x
