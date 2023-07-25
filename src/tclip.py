# coding=UTF-8
'''
Author: zhangyuanhang
LastEditors: zhangyuanhang
Date: 2022-11-21 22:05:38
LastEditTime: 2022-11-23 22:26:49
Description: 
'''
# coding:utf-8
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root_dir)
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0],True)
logical_devices = tf.config.list_logical_devices("GPU")
print(logical_devices)
import model.bert_tf2 as bert
import numpy as np
from transformers import TFCLIPVisionModel

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
        .shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #.batch(batch_size)
        .padded_batch(batch_size, padded_shapes=(
            {
                "input_ids":   [256],
                "input_mask":  [256],
                "image": [3, 224, 224],

            }))
    )


def create_text_encoder(max_seq_length, 
                    bert_config_file, 
                    is_training=False,
                    final_hidden_size=128,
                    ):
    # load bert config
    bert_config = bert.BertConfig.from_json_file(bert_config_file)
    # input
    input_ids  = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    # create bert model and load pretrain model
    bert_model = bert.BertModel(bert_config, is_training=is_training)

    bert_model.trainable = is_training
    all_encoder_layers = bert_model(input_ids, input_mask)
    #use [CLS] layer [first-last avg]
    bert_output = (all_encoder_layers[-0] + all_encoder_layers[-1])/2.0
    if final_hidden_size != 768:
        bert_output = tf.keras.layers.Dense(final_hidden_size)(bert_output)
    text_embedding = tf.reduce_mean(bert_output, 1)
    return tf.keras.models.Model(inputs=[input_ids, input_mask], outputs=text_embedding, name="text_encoder")



def create_image_encoder(
            pretrained_model="openai/clip-vit-base-patch32", 
            final_hidden_size=768,
            is_training=True):

    pixel_values = tf.keras.layers.Input(
        shape=(None, None, None), dtype=tf.float32, name="pixel_values"
    )
    vit_model = TFCLIPVisionModel.from_pretrained(pretrained_model)
    vit_model.trainable = is_training

    outputs = vit_model(pixel_values=pixel_values).pooler_output
    
    if final_hidden_size != 768:
        outputs = tf.keras.layers.Dense(final_hidden_size)(outputs)
    return tf.keras.models.Model(inputs=pixel_values, outputs=outputs, name="image_encoder")


class CLIP(tf.keras.Model):
    def __init__(self, text_encoder, image_encoder, temperature=1.0, **kwargs):
        super(CLIP, self).__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        text_embeddings  = text_encoder([features["input_ids"], features["input_mask"]])
        image_embeddings = vision_encoder(features["image"], training=training)
        return text_embeddings, image_embeddings

    def compute_loss(self, text_embeddings, image_embeddings):
        logits = (
            tf.matmul(text_embeddings, image_embeddings, transpose_b=True)
            / self.temperature
        )
        images_similarity = tf.matmul(
            image_embeddings, image_embeddings, transpose_b=True
        )
        text_similarity = tf.matmul(
            text_embeddings, text_embeddings, transpose_b=True
        )
        targets = tf.keras.activations.softmax(
            (text_similarity + images_similarity) / (2 * self.temperature)
        )
        text_loss = tf.keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        images_loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        return (text_loss + images_loss) / 2

    def compute_loss_v1(self, text_embeddings, image_embeddings):
        # 构造标签
        y_true = tf.eye(tf.shape(text_embeddings)[0])
        # 计算相似度
        nor_text_embeddings  = tf.math.l2_normalize(text_embeddings, axis=1)
        nor_image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)

        similarities = tf.matmul(nor_text_embeddings, nor_image_embeddings, transpose_b=True) * 20
        loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
        return tf.reduce_mean(loss)

    def train_step(self, features):
        with tf.GradientTape() as tape:
            # Forward pass
            text_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss_v1(text_embeddings, image_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        text_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss_v1(text_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


num_epochs = 60  # In practice, train for at least 30 epochs
batch_size = 128
bert_config_file = "/data2/zhangyuanhang/nlp_task/pretrain_model/distilbert_tf/bert_config.json"

vision_encoder = create_image_encoder(
    is_training=True
)
vision_encoder.summary()
text_encoder = create_text_encoder(
    max_seq_length=256, bert_config_file=bert_config_file, is_training=False, final_hidden_size=768
)
text_encoder.load_weights("./model_2l_6-10e_128b_768d/model.ckpt-4").expect_partial()
text_encoder.summary()
clip_model = CLIP(text_encoder, vision_encoder, temperature=0.05)


clip_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5)
)

class SaveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        while True:
            try:
                vision_encoder.save("./clip_model/vision_encoder")
                text_encoder.save("./clip_model/text_encoder")
                break
            except:
                print("save model to disk failure! retry!")
                
print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Batch size: {batch_size}")

train_dataset = get_dataset("../data/clip_train_data_short_000*.tfrecord", batch_size)
valid_dataset = get_dataset("../data/clip_train_data_short_0010.tfrecord", batch_size)

# Create a learning rate scheduler callback.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5
)
# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

model_save = SaveCallback()

history = clip_model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=[reduce_lr, early_stopping, model_save],
)
print("Training completed. Saving vision and text encoders...")
vision_encoder.save("./clip_model/vision_encoder")
text_encoder.save("./clip_model/text_encoder")
print("Models are saved.")

