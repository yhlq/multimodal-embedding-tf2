# coding=UTF-8
'''
Author: zhangyuanhang
LastEditors: zhangyuanhang
Date: 2022-10-28 22:01:47
LastEditTime: 2022-12-02 16:17:07
Description: 
'''
import requests
from PIL import Image
import hashlib
import numpy as np

class ImageProcess():
    def __init__(self,resize=224):
        self._mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self._std  = np.array([0.26862954, 0.26130258, 0.27577711])
        self.resize = resize

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def _image_center_crop(self, image, crop_size):
        width, height = image.size
        top = (height - crop_size[1]) // 2
        left = (width - crop_size[0]) // 2
        bottom = (height + crop_size[1]) // 2
        right = (width + crop_size[0]) // 2
        return image.crop((left, top, right, bottom))

    def _image_normalize(self, image):
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - self._mean[:, None, None]) / self._std[:, None, None]
        else:
            return (image - self._mean) / self._std

    def __call__(self, image, normalized=True):
        if isinstance(image, str):# and image.startswith('http'):
            image = Image.open(requests.get(image, stream=True, timeout=1, headers={'Connection':'close'}).raw)
        assert isinstance(image, Image.Image)
        return self._image_transform(image, normalized).tolist()

    def _image_transform(self, image, normalized=True):
        ''' 图像预处理: resize、center_crop、to_rgb、to_tensor、normalized'''
        assert isinstance(image, Image.Image)
        width, height = image.size
        _lamd = self.resize / min(width, height)
        _shape =  int(width * _lamd), int(height * _lamd)
        resized_image = image.resize(_shape,resample=Image.BICUBIC)  # resize
        cut_image = self._image_center_crop(resized_image, [self.resize, self.resize])
        if cut_image.mode != "RGB":
            print("not RGB image! mode = {} ".format(cut_image.mode))
        rgb_image = self._convert_image_to_rgb(cut_image)
        if not normalized:
            return np.array(rgb_image).transpose(2, 0, 1)
        # to tensor
        rgb_image = (np.array(rgb_image) / 255.0).transpose(2, 0, 1)
        # normalized
        return np.round(self._image_normalize(rgb_image) ,7)

    @staticmethod
    def convert_picid_to_url(pic_id):
        def _code(key, length=3):
            return hashlib.md5(key.encode("utf-8")).hexdigest()[:length]
        urlKey, sw = _code(f"{pic_id}water"), _code(f"{pic_id}severno") 
        return 'https://img.haohaozhu.cn/App-imageShow/o_phone/' + urlKey + '/' + pic_id + '?sw=' + sw + '&w=230&h=23&iv=1'


if __name__ == "__main__":
    import time
    image = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = ImageProcess.convert_picid_to_url("938e920hs0hs00000nygyy4")

    print(image)
    image = Image.open(requests.get(image, stream=True).raw)
    iclient = ImageProcess()
    st = time.time()
    ret = iclient(image,normalized=False)
    print(1000*(time.time() - st))
    print(ret)
    print(ret[0][0][0:20], np.array(ret).shape, sep="\n")
    

