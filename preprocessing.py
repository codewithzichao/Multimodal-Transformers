import os
import numpy as np
import pandas as pd
import codecs
from PIL import Image
import json
import re
from collections import defaultdict
from torchvision.transforms import transforms

# 去除emoji
def demoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U00010000-\U0010ffff"
                               "]+", flags=re.UNICODE)
    return (emoji_pattern.sub(r'', text))


def process_text(text_path,output_path):
    data_df=pd.read_csv(text_path,sep=",",encoding="utf-8",header=0,names=["id","imagename","captions"])
    data_df["captions"]=data_df["captions"].apply(lambda x:demoji(x))
    label=["Not_troll" if "Not_troll" in item else "troll" for item in data_df["imagename"]]
    data_df["label"]=label
    data_df.to_csv(output_path,sep="\t",encoding="utf-8",index=False)

    label_freq={"Not_troll":0,"troll":0}
    for item in data_df["imagename"]:
        if "Not_troll" in item:
            label_freq["Not_troll"]+=1
        else:
            label_freq["troll"]+=1

    with codecs.open("data/label.json", "w", encoding="utf-8") as f:
        json.dump(label_freq,f,ensure_ascii=True)


def zip_image_text_label(image_path,text_path):

    data=[]
    my_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), \
        transforms.ToTensor(), \
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    text_df=pd.read_csv(text_path,sep="\t",encoding="utf-8",header=0)
    for idx,item in text_df.iterrows():
        temp=[]

        image_name=item["imagename"]
        image_item=Image.open(image_path+"/"+image_name)
        image_item=image_item.convert("RGB")
        image_item=my_transform(image_item)

        temp.append(image_item.numpy())
        temp.append(item["captions"])
        temp.append(item["label"])

        data.append(temp)

    return np.array(data)

def zip_image_text(image_path,text1_path):

    data=[]
    my_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), \
        transforms.ToTensor(), \
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    text_df=pd.read_csv(text1_path,sep=",",encoding="utf-8",header=0)
    print(text_df.head(10))
    for idx,item in text_df.iterrows():
        temp=[]

        image_name=item["imagename"]
        image_item=Image.open(image_path+"/"+image_name)
        image_item=image_item.convert("RGB")
        image_item=my_transform(image_item)

        temp.append(image_item.numpy())
        temp.append(demoji(item["captions"]))
        temp.append(image_name.strip().split(".")[0])

        data.append(temp)

    return np.array(data)


def get_longest_length(data_path):
    data_df=pd.read_csv(data_path,sep="\t",encoding="utf-8",header=0)
    ans=0
    for item in data_df["captions"]:
        ans=max(ans,item.__len__())

    return ans



if __name__=="__main__":
    base_path="/Users/codewithzichao/Desktop/competitions/meme_EACL2021"
    total_data_path=base_path+"/data/train_captions.csv"
    picture_path=base_path+"/data/uploaded_tamil_memes"
    output_path=base_path+"/data/data.csv"

    process_text(total_data_path,output_path)

    data=zip_image_text_label(picture_path,output_path)

    print(data.shape) #(2300, 3)
    image_list=os.listdir(picture_path)
    print(image_list.__len__())

    ans=get_longest_length(output_path)
    print(ans) # 最长有545

