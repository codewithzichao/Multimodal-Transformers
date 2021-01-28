import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from preprocessing import zip_image_text_label
from train import K_fold_training
from args import MyArgs
from transformers import ElectraTokenizer,XLMRobertaTokenizer,XLMRobertaModel
import numpy as np

args = MyArgs()

# 数据路径
base_path = args.base_path
image_data_path = base_path + "/data/uploaded_tamil_memes"
text_data_path = base_path + "/data/data.csv"
origin_test_pic_path=base_path+"/test_data/test_img"
pseudo_label_path=base_path+"/pseudo.csv"

# 模型路径
resnet_path = base_path + "/pretrained_weights/resnet/resnet152-b121ed2d.pth"
bert_path=base_path+"/pretrained_weights/xlm-roberta-base/"
bert_vocab_path=base_path+"/pretrained_weights/xlm-roberta-base/sentencepiece.bpe.model"

# 获取数据
all_zipped_data_origin = zip_image_text_label(image_data_path, text_data_path)
label2idx = {"Not_troll": 0, "troll": 1}
tokenizer=XLMRobertaTokenizer(vocab_file=bert_vocab_path)
base_save_path = base_path + "/ckpt"
dropout_rate = 0.2
num_class = 2
eval_step_interval = 200
base_writer_path = base_path + "/logfile"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K折训练
print("start training!")

K_fold_training(model_name=args.model_name, data_np=all_zipped_data_origin, fold_num=args.fold_num, label2idx=label2idx, \
                tokenizer=tokenizer, resnet_path=resnet_path, bert_path=bert_path, dropout_rate=dropout_rate, \
                num_class=num_class, epochs=args.epochs, batch_size=args.batch_size, accum_num=args.accum_num,\
                base_save_path=base_save_path, max_norm=args.max_norm, eval_step_interval=eval_step_interval,\
                base_writer_path=base_writer_path, device=device)

