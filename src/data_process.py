#coding:utf-8
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
import collections
import tensorflow as tf
from utils import tokenization
import os
import json

class ConvertTextToIds():
    def __init__(self, vocab_file="/data2/zhangyuanhang/nlp_task/pretrain_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt", 
                    do_lower_case=True, 
                    max_seq_length=256
        ):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_length = max_seq_length

    def __call__(self, text):
        return self.text_to_ids(text)
        
    def text_to_ids(self, text_a, text_b=None):
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
    def __init__(self, vocab_file, do_lower_case, max_seq_length=128, label_file=""):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_length = max_seq_length
        self.label_map = json.load(open(label_file))
        self.label_id = dict(zip(self.label_map.values(), self.label_map.keys()))      
        self.max_label_num = 10

    def text_to_ids(self, text_a, text_b=None, label=''):
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
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids)   == self.max_seq_length
        assert len(input_mask)  == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        
        label_id = self.label_map[label] if label else -1

        return (input_ids, input_mask, segment_ids, [label_id])

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

    def file_based_convert_to_features(self, input_file, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        sentence_id = 0
        writer = tf.io.TFRecordWriter(output_file)
        with open(input_file, "r") as pf:
            for line in pf:
                #data = line.strip().split("|||")
                #if len(data) != 3:  continue
                data = json.loads(line.strip())
                query = data.get("title")
                doc   = data.get("remark")
                #query, doc, lables = data
                #obj_id, title, remark, admin_tag, cate_first, cate_second = data
                if (not query or query == "None") and (not doc or doc == "None"):
                    continue
                sa_input_ids, sa_input_mask, *_ = self.text_to_ids(text_a=query)
                sb_input_ids, sb_input_mask, *_ = self.text_to_ids(text_a=doc)
                for input_ids, input_mask in [(sa_input_ids, sa_input_mask), (sb_input_ids, sb_input_mask)]:
                    features = collections.OrderedDict()
                    features["input_ids"] = create_int_feature(input_ids)
                    features["input_mask"] = create_int_feature(input_mask)
                    features["sentence_id"] = create_int_feature([sentence_id])
                    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(tf_example.SerializeToString())
                sentence_id += 1
        print("sentence_id : {} ".format(sentence_id))
        writer.close()


if __name__ == "__main__":
    init_checkpoint = "/data2/zhangyuanhang/nlp_task/pretrain_model/chinese_roberta_wwm_ext_L-12_H-768_A-12"
    train_data = "../data/"
    cttt = ConvertTextToTFRecord(vocab_file=os.path.join(init_checkpoint, "vocab.txt"), 
                                    do_lower_case=True, 
                                    max_seq_length=256,
                                    label_file=os.path.join(train_data, "labels.json")
    )
    print(cttt.text_to_ids("新家装修后如何处理甲醛超标？"))
    ## convert train data to tfrecord
    cttt.file_based_convert_to_features(os.path.join(train_data, "sts_train_data_all.txt"), os.path.join(train_data, "sts_train_data_all.tfrecord"))
    #cttt.tfrecord_decode(os.path.join(train_data, "sub_category.tfrecord")) # for test
    ## convert test data to tfrecord
    #cttt.file_based_convert_to_features(os.path.join(train_data, "test.txt"), os.path.join(train_data, "test_tfrecord"))
    #cttt.tfrecord_decode(os.path.join(train_data, "test_tfrecord")) # for test
