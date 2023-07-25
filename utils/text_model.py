# coding=UTF-8
'''
Author: zhangyuanhang
LastEditors: zhangyuanhang
Date: 2022-12-06 11:18:22
LastEditTime: 2022-12-06 11:40:17
Description:
'''
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
import tensorflow as tf
import numpy as np
from common.data_process import ConvertTextToIds
from data_helper import ConvertTextToIds
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

class TextEncoder():
    '''vit 直接加载saved_model模型 进行推理'''
    def __init__(self, model_dir, vocab_file):
        # set gpu
        USING_GPU_INDEX = 0
        gpus = tf.config.list_physical_devices('GPU')
        print(gpus, '*******************************')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[USING_GPU_INDEX],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        # load model
        self.unsigmodel = tf.saved_model.load(model_dir)
        self.text_to_ids = ConvertTextToIds(vocab_file=vocab_file)

    def inference(self, text, normalize=True):
        input_ids, input_mask, _ = self.text_to_ids(text)
        _feed_dict = {
            "input_ids": tf.constant([input_ids]),
            "input_mask":tf.constant([input_mask]),
        }
        text_model = self.unsigmodel.signatures['serving_default']
        text_embedding = text_model(**_feed_dict)["tf_op_layer_Mean"][0].numpy()
        if normalize:
            text_embedding = text_embedding/np.linalg.norm(text_embedding)
            text_embedding = [round(i, 6) for i in text_embedding.tolist()] ## 转化为python float类型
        return text_embedding
