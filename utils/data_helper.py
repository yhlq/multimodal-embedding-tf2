
# coding=UTF-8
'''
Author: zhangyuanhang
LastEditors: zhangyuanhang
Date: 2022-11-22 14:28:03
LastEditTime: 2022-11-26 18:22:08
Description: 
'''
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
import collections
import tensorflow as tf
from utils import tokenization
from utils.image_process import ImageProcess
import numpy as np
import os
import json
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class ConvertTextToIds():
    '''使用bert分词 对中文进行分词、转码'''
    def __init__(self, vocab_file="../data/vocab.txt", 
                    do_lower_case=True, 
                    max_seq_length=256
        ):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_length = max_seq_length

    def __call__(self, text, padding=True):
        return self.text_to_ids(text, padding=padding)
        
    def text_to_ids(self, text_a, text_b=None, padding=True):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b=None
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        if padding:
            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids)   == self.max_seq_length
            assert len(input_mask)  == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            
        return input_ids, input_mask, segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class ConvertTextToTFRecord():
    def __init__(self, vocab_file, do_lower_case, max_seq_length=128):
        self.max_seq_length = max_seq_length
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case

    def file_based_convert_to_features(self, input_file, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        sentence_id = 0
        writer = tf.io.TFRecordWriter(output_file)
        with open(input_file, "r") as pf:
            for line in pf:             
                data = json.loads(line.strip())
                query = data.get("title")
                doc   = data.get("remark")
                if (not query or query == "None") and (not doc or doc == "None"):
                    continue
                q_input_ids, q_input_mask, _ = ConvertTextToIds(vocab_file=self.vocab_file,
                                                do_lower_case=self.do_lower_case,
                                                max_seq_length=self.max_seq_length
                                                )(query)
                d_input_ids, d_input_mask, _ = ConvertTextToIds(vocab_file=self.vocab_file,
                                                do_lower_case=self.do_lower_case,
                                                max_seq_length=self.max_seq_length
                                                )(doc)
                for input_ids, input_mask in [(q_input_ids, q_input_mask), (d_input_ids, d_input_mask)]:
                    features = collections.OrderedDict()
                    features["input_ids"] = create_int_feature(input_ids)
                    features["input_mask"] = create_int_feature(input_mask)
                    features["sentence_id"] = create_int_feature([sentence_id])
                    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(tf_example.SerializeToString())
                sentence_id += 1
        print("sentence_id : {} ".format(sentence_id))
        writer.close()


class TextImageDataMaker():
    def __create_example(self, text, image_path, input_is_picid=False):
        # 处理文本特征
        input_ids, input_mask, _ = ConvertTextToIds()(text, padding=False)
        features = collections.OrderedDict()
        # 处理图像特征
        if input_is_picid:
            image_path = ImageProcess().convert_picid_to_url(image_path)
        raw_image = np.array(ImageProcess()(image_path)).astype(np.float32).tostring()
        
        features["input_ids"]  = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_ids)))
        features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_mask)))
        features["raw_image"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_image]))

        return tf.train.Example(features=tf.train.Features(feature=features))
        
    def write_tfrecords(self, input_file, output_file_prefix, input_is_picid=False, single_max_num=20000):
        def to_tfrecords(output_file_prefix, example_list, index):
            with tf.io.TFRecordWriter("{}_1{:0>4d}.tfrecord".format(output_file_prefix, index)) as writer:
                for example in example_list:
                    writer.write(example.SerializeToString())
            return 
        #writer = tf.io.TFRecordWriter(output_file)
        _index = 0
        example_list = []
        with open(input_file, "r") as pf:
            for line in pf:
                data = json.loads(line.strip())
                text = data['text']
                image_path = data['image_path']
                _index += 1
                print(text, image_path, sep="\t")
                try:
                    example = self.__create_example(text, image_path, input_is_picid)
                except Exception as e:
                    print(e)
                    continue
                example_list.append(example)
                if len(example_list) >= single_max_num:
                    to_tfrecords(output_file_prefix, example_list, index=_index // single_max_num)
                    example_list = []
        if example_list:
            to_tfrecords(output_file_prefix, example_list, index=_index // single_max_num + 1)
            example_list = []

    
    def __call__(self, input_file, output_file, input_is_picid=False):
        self.write_tfrecords(input_file, output_file, input_is_picid)
    

def get_dataset(file_pattern, batch_size=1):
    feature_description = {
        "input_ids": tf.io.VarLenFeature(tf.int64),
        "input_mask": tf.io.VarLenFeature(tf.int64),
        "raw_image": tf.io.FixedLenFeature([], tf.string),
    }

    def read_example(example):
        features = tf.io.parse_single_example(example, feature_description)
        raw_image = features.pop("raw_image")
        features["image"] = tf.reshape(tf.io.decode_raw(raw_image, tf.float32), [3, 224, 224])  
        features["input_ids"]    = tf.sparse.to_dense(features["input_ids"])
        features["input_mask"]   = tf.sparse.to_dense(features["input_mask"])
        return features
        
    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(
            read_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False,
        )
        #.shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
    )



if __name__ == "__main__":
    TextImageDataMaker()("/data2/zhangyuanhang/data/hhz_image_caption.txt", "/data3/hhz_image_caption_tmp/clip_train_data_hhz")
    exit()

    count = 0
    for i in get_dataset("clip_train_data_short_0000.tfrecord"):
        count += 1
    print(count)