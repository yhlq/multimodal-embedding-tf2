# coding:utf-8
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
import tensorflow as tf
import model.bert_tf2 as bert
import numpy as np


feature_description = {
    "input_ids": tf.io.VarLenFeature(tf.int64),
    "input_mask": tf.io.VarLenFeature(tf.int64),
    #"segment_ids": tf.io.VarLenFeature(tf.int64),
    "sentence_id": tf.io.FixedLenFeature([], tf.int64),
}

def read_example(example):
    example      =  tf.io.parse_single_example(example, feature_description)
    input_ids    = tf.sparse.to_dense(example["input_ids"])
    input_mask   = tf.sparse.to_dense(example["input_mask"])
    sentence_id    = example.get("sentence_id")
    return (input_ids, input_mask), 1

def get_dataset(file_pattern, batch_size):
    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(
            read_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False,
        )
        #.shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #.batch(batch_size)
        .padded_batch(batch_size, padded_shapes=(([-1],[-1]), []))
        #.padded_batch(batch_size, padded_shapes=(
        #    {
        #        "input_ids":   [-1],
        #        "input_mask":  [-1],
        #        #"input_ids_p": [-1],
        #        #"input_mask_p":[-1],
        #    }, []))
        
    )

class InfoNCELoss(tf.keras.losses.Loss):
    def __init__(self):
        super(InfoNCELoss, self).__init__()

    # Compute loss
    def ttcall(self, y_true, y_pred):
        query_embedding = tf.math.l2_normalize(y_pred[::2], axis=1)
        doc_embedding   = tf.math.l2_normalize(y_pred[1::2], axis=1)
        #batch_size = query_embedding.shape[0]
        batch_size = tf.cast(tf.shape(query_embedding)[0], tf.int32)
        #单位矩阵作为label
        y_true = tf.eye(batch_size)
        similarities = tf.matmul(query_embedding, doc_embedding, transpose_b=True)
        loss = tf.keras.losses.categorical_crossentropy(y_true, similarities)
        return tf.reduce_mean(loss)

    # Compute loss
    def call(self, y_true, y_pred):
        # 构造标签
        idxs = tf.range(0, tf.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.float32)
        # 计算相似度
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
        similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
        similarities = similarities * 20
        loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
        return tf.reduce_mean(loss)
        

def create_text_encoder(max_seq_length, 
                    bert_config_file, 
                    init_checkpoint, 
                    is_training=True,
                    final_hidden_size=768,
                    ):
    # load bert config
    bert_config = bert.BertConfig.from_json_file(bert_config_file)
    # input
    input_ids  = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    # create bert model and load pretrain model
    bert_model = bert.BertModel(bert_config, is_training=is_training)
    
    #bert_model.fit([], epochs=0)
    #bert_model.load_weights(init_checkpoint)
    if os.path.exists(init_checkpoint) and is_training:
        print("load bert pretrain model!!!!!!!!!")
        bert_model.load_weights(init_checkpoint)
    else:
        print("bert pretrain model not exists!!!!!!!!!")
    all_encoder_layers = bert_model(input_ids, input_mask)
    #use [CLS] layer [first-last avg]
    bert_output = (all_encoder_layers[-0] + all_encoder_layers[-1])/2.0
    if final_hidden_size != 768:
        bert_output = tf.keras.layers.Dense(final_hidden_size)(bert_output)
    text_embedding = tf.reduce_mean(bert_output, 1, name="text_encoder")
    return tf.keras.models.Model(inputs=[input_ids, input_mask], outputs=text_embedding, name="text_encoder")


def main(TrainParams):
    set_gpu()
    # load train data
    print("load train data ...")
    train_dataset = get_dataset(TrainParams.train_tfrecord_file, TrainParams.batch_size)
    train_data_nums = 0
    for data in train_dataset:
        train_data_nums += 1
    print("train_data_nums : {} ".format(train_data_nums))

    print("build model ...")
    text_encoder = create_text_encoder(max_seq_length=TrainParams.max_seq_length,
                            bert_config_file=TrainParams.bert_config_file,
                            init_checkpoint=TrainParams.init_checkpoint,
                            is_training=TrainParams.is_training
    )
    print("load pretrain model ...")
    '''
    text_encoder.load_weights(TrainParams.init_checkpoint)
    for layer in text_encoder.layers:
        for w in layer.weights:
            print(w.name, w.shape)
    tf.keras.utils.plot_model(text_encoder, show_shapes=True)
    exit()
    for ii in text_encoder.get_weights():
        print(ii)
    #print(text_encoder.get_weights())
    tf.keras.utils.plot_model(text_encoder, show_shapes=True)
    exit()
    '''
    #text_encoder.load_weights(TrainParams.init_checkpoint)
    text_encoder.load_weights("model_2l_5e_128b_768d/text_encoder_model.h5", by_name=True)
    if TrainParams.is_training:
        class Evaluator(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                while True:
                    try:
                        text_encoder.save_weights(f'./model/model.ckpt-{epoch}')
                        text_encoder.save_weights("./model/text_encoder_model.h5")
                        break
                    except:
                        print("save model to disk failure! retry!")
        text_encoder.compile(optimizer=tf.keras.optimizers.Adam(TrainParams.lr), loss=InfoNCELoss())

        evaluator = Evaluator()
        # print model
        text_encoder.summary()

        print("start training ... ")
        text_encoder.fit(
            train_dataset,
            epochs=TrainParams.epochs,
            callbacks=[evaluator,tf.keras.callbacks.TensorBoard(log_dir=f"logs/text_encoder")],
        )
        text_encoder.save_weights("./model/text_encoder_model.h5")
    else:
        print("do predict test ....")
        # load model
        text_encoder.load_weights("./model/text_encoder_model.h5")
        ####### test predict
        # fake input
        input_ids = np.array([[102,99]*128])
        input_mask = np.array([[1,1]*128])

        ret = text_encoder([input_ids, input_mask])
        print(ret)


def set_gpu():
    USING_GPU_INDEX = 0
    gpus = tf.config.list_physical_devices('GPU')
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
        '''
        try:
            tf.config.set_logical_device_configuration(
                gpus[USING_GPU_INDEX],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
        '''

class TrainParams():
    #params
    is_training = True
    max_seq_length = 256
    batch_size = 128
    lr = 2e-5
    epochs = 5
    # bert config
    #init_checkpoint = "/data2/zhangyuanhang/nlp_task/pretrain_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt"
    #bert_config_file = "/data2/zhangyuanhang/nlp_task/pretrain_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"
    #train_tfrecord_file = "/data2/zhangyuanhang/nlp_task/multimodal_embedding/data/sts_train.tfrecord"
    train_tfrecord_file = "/data2/zhangyuanhang/nlp_task/multimodal_embedding/data/sts_train_data_all.tfrecord"
    init_checkpoint = "/data2/zhangyuanhang/nlp_task/pretrain_model/distilbert_tf/bert_model.ckpt"
    bert_config_file = "/data2/zhangyuanhang/nlp_task/pretrain_model/distilbert_tf/bert_config.json"



if __name__ == "__main__":
    main(TrainParams)