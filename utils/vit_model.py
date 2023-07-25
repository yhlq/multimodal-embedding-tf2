# coding=UTF-8
'''
Author: zhangyuanhang
LastEditors: zhangyuanhang
Date: 2022-10-28 21:57:59
LastEditTime: 2022-12-06 11:17:30
Description: vit模型推理代码
'''
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
import tensorflow as tf  
import numpy as np
from utils.image_process import ImageProcess
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class VitPredict():
    '''vit 直接加载saved_model模型 进行推理'''
    def __init__(self, model_dir):
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

    def inference(self, image):
        if isinstance(image, list):
            _feed_dict = {"pixel_values": tf.constant([image]),}
        else:
            _feed_dict = {"pixel_values": tf.constant([ImageProcess()(image=image)]),}
        
        sst = time.time()
        vit_model = self.unsigmodel.signatures['serving_default']
        result = vit_model(**_feed_dict)['tfclip_vision_model'].numpy()
        print("cost:{:.4f}".format(1000*(time.time()- sst)))
        # normalize
        norm_vec = result[0]/np.linalg.norm(result[0])
        return np.round(norm_vec,6)


if __name__ == "__main__":
    export_dir = "clip_model/d768v1/vision_encoder/"
    vit_client = VitPredict(model_dir=export_dir)
    image = "https://img.haohaozhu.cn/App-imageShow/o_phone/5fd/37b7f22io1w000000op3y342j1u?sw=a40&iv=1"
    for _ in range(10):
        st = time.time()
        image_vec = vit_client.inference(image)
        print(1000*(time.time()-st))
    #print(image_vec)