import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from preprocessing import demoji,zip_image_text
from gen_data import MyTestDataset,MyTestDataset1
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from model import MyModel
from tqdm import tqdm
import numpy as np
import codecs
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据路径
base_path = "/data-tmp/meme-EACL2021-xlmr"
picture_path = base_path + "/test_data/test_img"
text_path = base_path + "/test_data/test_captions.csv"

# 预测数据存放路径
pred_data_base_path = base_path + "/pred_data"

# 模型路径
ckpt_path = base_path + "/ckpt"
roberta_path = base_path + "/xlm-roberta-base/"
roberta_vocab_path = roberta_path + "sentencepiece.bpe.model"
resnet_path = base_path + "/resnet/resnet152-b121ed2d.pth"
roberta_tokenizer = XLMRobertaTokenizer(vocab_file=roberta_vocab_path)

# 测试集数据
test_data_origin = zip_image_text(picture_path, text_path)
test_data = MyTestDataset(test_data_origin, roberta_tokenizer)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# 记得改回来
label2idx = {"not_troll": 0, "troll": 1}
idx2label = {0: "not_troll", 1: "troll"}

# 模型

#######################################################################################################
# 构造inference函数
def test_and_to_csv(test_loader, pred_data_base_path, num=1):

    pred_data_base_path = pred_data_base_path + "/%d_pred.csv" % num
    my_model = MyModel("xlm-roberta", resnet_path,roberta_path, 0.2, 2)
    my_model.load_state_dict(torch.load(ckpt_path+ "/%d_best.ckpt"%num, map_location=device)["model"], strict=False)
    my_model.to(device)

    y_preds = []
    image_names = []
    probs = []

    my_model.eval()
    with torch.no_grad():
        for idx, batch_data in tqdm(enumerate(test_loader)):
            image, input_ids, attention_mask, image_name = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            logits = my_model(image.to(device), input_ids.to(device), attention_mask.to(device))

            probs.append(F.softmax(logits.cpu(), dim=-1).numpy())
            logits = logits.cpu().numpy()
            image_names.append(image_name[0])

            #print(logits)
            for item in logits:
                y_preds.append(idx2label[np.argmax(item)])

    str_format = "{imagename},{label}\n"
    with codecs.open(pred_data_base_path, "w", encoding="utf-8") as f:
        f.write("imagename,label\n")

        for idx, item in tqdm(enumerate(y_preds)):
            f.write(str_format.format(imagename=image_names[idx], label=y_preds[idx]))

        f.flush()
        f.close()

    probs = np.array(probs)

    del my_model
    return probs, image_names


def merge_all_test_and_to_csv(probs_list, image_names,pred_data_base_path):

    prob1, prob2, prob3, prob4, prob5 = probs_list[0], probs_list[1], probs_list[2], probs_list[3], probs_list[4]
    avg_prob = (prob1 + prob2 + prob3 + prob4 + prob5) / 5.0

    pred_data_base_path = pred_data_base_path + "/mean_pred.csv"
    y_preds = []
    for item in avg_prob:
        y_preds.append(idx2label[np.argmax(item)])


    str_format = "{imagename},{label}\n"
    with codecs.open(pred_data_base_path, "w", encoding="utf-8") as f:
        f.write("imagename,label\n")

        for idx, item in tqdm(enumerate(y_preds)):
            f.write(str_format.format(imagename=image_names[idx], label=y_preds[idx]))

        f.flush()
        f.close()

#######################################################################################################
# 加入伪标签
def merge_all_test_and_to_csv_pseudo(probs_list, image_names,output_path):

    prob1, prob2, prob3, prob4, prob5 = probs_list[0], probs_list[1], probs_list[2], probs_list[3], probs_list[4]
    avg_prob = (prob1 + prob2 + prob3 + prob4 + prob5) / 5.0

    y_preds = []

    assert len(avg_prob)==len(image_names)

    for i in range(len(avg_prob)):
        if np.max(avg_prob[i])>0.99:
            assert test_data_origin[i][-1]==image_names[i]
            y_preds.append({"imagename":image_names[i],"captions":test_data_origin[i][1],"label":idx2label[np.argmax(avg_prob[i])]})
        #y_preds.append(idx2label[np.argmax(item)])

    str_format = "{imagename}\t{captions}\t{label}\n"
    with codecs.open(output_path, "w", encoding="utf-8") as f:
        f.write("imagename\tcaptions\tlabel\n")

        for idx, item in tqdm(enumerate(y_preds)):
            f.write(str_format.format(imagename=item["imagename"],captions=item["captions"],label=item["label"]))

        f.flush()
        f.close()

#######################################################################################################

print("start testing!")

print("start testing 1 model!")
probs1,images1=test_and_to_csv(test_loader,pred_data_base_path,num=1)
print("start testing 2 model!")
probs2,images2=test_and_to_csv(test_loader,pred_data_base_path,num=2)
print("start testing 3 model!")
probs3,images3=test_and_to_csv(test_loader,pred_data_base_path,num=3)
print("start testing 4 model!")
probs4,images4=test_and_to_csv(test_loader,pred_data_base_path,num=4)
print("start testing 5 model!")
probs5,images5=test_and_to_csv(test_loader,pred_data_base_path,num=5)

print("start testing avg model!")
probs_list=[probs1,probs2,probs3,probs4,probs5]
#pseudo_path=base_path+"/pseudo.csv"
merge_all_test_and_to_csv(probs_list,images1,pred_data_base_path)
#merge_all_test_and_to_csv_pseudo(probs_list,images1,pseudo_path)

print("end testing!")
print("finished all testing!")

#######################################################################################################

