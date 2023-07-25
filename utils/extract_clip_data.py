# coding=UTF-8
'''
Author: zhangyuanhang
LastEditors: zhangyuanhang
Date: 2022-11-22 21:23:39
LastEditTime: 2023-01-13 16:12:45
Description: 
'''
import os
import sys
import json
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
import re
import requests
from bs4 import BeautifulSoup
from utils.image_process import ImageProcess

def fetch_data_from_html(url="https://new.qq.com/rain/a/20210909A0CBWK00"):
    data = requests.get(url)
    soup = BeautifulSoup(data.text, 'lxml')
    alldata = soup.find_all('p')
    text_list = []
    plants_list = []
    for adata in alldata:
        text = adata.get_text().strip()
        if text:
            tmp_text = re.sub("\d+\.","", text)
            if tmp_text != text:
                plants_list.append(tmp_text)
            text_list.append(tmp_text)
            continue
        img_url = "https://" + adata.img.get('src').replace("//", "")
        
        with open("plants_list", "a+") as pf:
            for text in text_list:
                tmp = json.dumps({"text":text, "image_path":img_url}, ensure_ascii=False)
                pf.write(tmp + "\n")
        text_list = []
    json.dump(plants_list, open("plants_name_list.json", "w"), ensure_ascii=False, indent=4)
    

def loadkeywords(filename="/data2/zhangyuanhang/nlp_task/download_corpus/keywords.json"):
    data = json.load(open(filename, "r"))
    keywords = []
    for cate1 in data:
        for cate2 in data[cate1]:
            keywords.extend(data[cate1][cate2])
    return data, keywords

def extract_data():
    output_file = open("clip_train_data.txt", "a+")

    dir = "/data2/zhangyuanhang/data/open_img_json_2.3M"
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        with open(file_path, "r") as pf:
            for line in pf:
                data = json.loads(line.strip())
                image_path = data['ImageUrl']
                image_query = data['ImageQuery']
                for text in image_query:
                    tmp = json.dumps({"text":text, "image_path":image_path}, ensure_ascii=False)
                    output_file.write(tmp + "\n")
                    
    output_file.close()


def split_data(filename="../data/clip_train_data.txt"):
    _, keywords = loadkeywords()
    plants = json.load(open("plants_name_list.json", "r"))
    keywords_pattern = re.compile("|".join(keywords+plants))
    count = 0
    deco_data = open("deco_data.txt", "a+")
    not_deco_data = open("not_deco_data.txt", "a+")

    with open(filename, "r") as pf:
        for line in pf:
            data = json.loads(line.strip())
            text = data["text"]
            if "绿箩" in text:  text = text.replace("绿箩", "绿萝")
            #image_path = data["image_path"]
            hit_keywords = re.findall(keywords_pattern, text)
            if not hit_keywords:   
                not_deco_data.write(line.strip() + "\n")
                continue
            deco_data.write(line.strip() + "\n")
            count += 1
    print(count)


def extract_hhz_image(input_file):
    outputfile = open("hhz_image_caption.txt", "a+")
    with open(input_file, "r") as pf:
        for line in pf:
            data = json.loads(line.strip())
            title  = data['title']
            remark = data['remark']
            text = ""
            if title and title != "None":
                text += title
            if remark and remark != "None":
                text += remark
            if not text:
                continue
            pic_list = json.loads(data['pic_list'])
            image_path = ImageProcess().convert_picid_to_url(pic_list[0]["pic_id"])
            outputfile.write(json.dumps({"text":text, "image_path":image_path}, ensure_ascii=False) + "\n")
    outputfile.close()



#split_data()
fetch_data_from_html()
#extract_hhz_image("/data2/zhangyuanhang/data/image_caption.txt")
