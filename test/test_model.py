# coding=UTF-8
'''
Author: zhangyuanhang
LastEditors: zhangyuanhang
Date: 2022-11-23 13:50:37
LastEditTime: 2022-12-02 22:01:52
Description: 
'''
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0],True)
logical_devices = tf.config.list_logical_devices("GPU")
print(logical_devices)
from vit_model import VitPredict
from utils.image_process import ImageProcess
from data_helper import ConvertTextToIds
import numpy as np
import requests
import json

def image_encoder_http(image_url):
    data = {
        "image":image_url,
    }
    headers = {'Content-Type': 'application/json','Connection': 'close'}
    response = requests.post(url='http://0.0.0.0:3155/vit_embedding/', headers=headers, data=json.dumps(data))
    return response.json()['ret']['embedding']

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def text_encoder_model(model_dir="./clip_model/d768v1/text_encoder"):
    text_encoder = tf.saved_model.load(model_dir)
    return text_encoder

def vision_encoder_model(model_dir="./clip_model/d768v1/vision_encoder"):
    vision_encoder = tf.saved_model.load(model_dir)
    return vision_encoder

def vision_encoder_client(model_dir="./clip_model/d768v1/vision_encoder"):
    vision_encoder_client = VitPredict(model_dir=model_dir)
    return vision_encoder_client

## create model
vision_encoder = vision_encoder_client()
text_encoder   = text_encoder_model()


### test data
image_url = "https://i1.jiajuol.com/0/newphoto/20150911/16/55f297e928e40.jpg!l"
image_url = "http://p3.img.360kuai.com/t01c434e06a1a65bd7f.jpg?size=640x562"
image_url = "https://img.haohaozhu.cn/App-imageShow/o_nphone/933/d31ca30u00qh0F700r5yh97738bp?iv=1&w=750&h=661"
image_url = "https://img.haohaozhu.cn/App-imageShow/o_phone/949/2e0bc215o1qi0HE00q9yaoj1iclq?sw=31b&w=230&h=23&iv=1"
image = ImageProcess()(image_url)
text_list = ["简约美式厨房", "开放酒柜", "卧室", "客厅", "厨房", "燃气灶", "抽油烟机", "沙发", "调料盒"]
text_list = ["水杯", "开放酒柜", "卧室", "调料盒", "厨房", "燃气灶", "抽油烟机", "沙发", "玻璃门", "马桶", "厕所", "卫生间"]


## 计算图片向量
#image_embeddings = vision_encoder([image]) ## 直接加载模型
#image_embeddings = vision_encoder.inference(image_url) ## 使用client
image_embeddings = image_encoder_http(image_url) ##  调用远程接口

print(image_embeddings)

## 计算文本向量
input_ids_list = []
input_mask_list = []
for text in text_list:
    input_ids, input_mask, _ = ConvertTextToIds()(text)
    input_ids_list.append(input_ids)
    input_mask_list.append(input_mask)
text_embeddings  = text_encoder([input_ids_list, input_mask_list])


## 文本相似性
for i in range(len(text_list)):
    query = text_list[i]
    query_embedding = text_embeddings[i]
    for j in range(i, len(text_list)):
        doc = text_list[j]
        doc_embedding = text_embeddings[j]
        print(query, doc, cos_sim(query_embedding, doc_embedding), sep="\t")

## 文本 & 图片相似性
for i in range(len(text_list)):
    print(text_list[i], cos_sim(text_embeddings[i], image_embeddings), sep="\t")

