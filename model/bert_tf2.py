# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import numpy as np
import tensorflow as tf


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=16,
            initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
            max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(tf.keras.Model):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True)

    label_embeddings = tf.Variable(...)
    pooled_output = model(input_ids, input_mask, token_type_ids) # return pooled_output by default
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """
    def __init__(self, config, is_training, name="bert"):
        """Constructor for BertModel.

        Args:
        config: `BertConfig` instance.
        is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
        name: (optional) Keras model name. Defaults to "bert".

        Raises:
        ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        super(BertModel, self).__init__(name=name)
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.embedding_lookup = embedding_lookup(name="embeddings",
                                            vocab_size=config.vocab_size,
                                            embedding_size=config.hidden_size,
                                            initializer_range=config.initializer_range,
                                            word_embedding_name="word_embeddings")
        self.embedding_postprocessor = embedding_postprocessor(name="embeddings",
                                                        use_token_type=True,
                                                        token_type_vocab_size=config.type_vocab_size,
                                                        token_type_embedding_name="token_type_embeddings",
                                                        use_position_embeddings=True,
                                                        position_embedding_name="position_embeddings",
                                                        initializer_range=config.initializer_range,
                                                        max_position_embeddings=config.max_position_embeddings,
                                                        dropout_prob=config.hidden_dropout_prob)
        self.transformer_model = transformer_model(name="encoder",
                                            hidden_size=config.hidden_size,
                                            num_hidden_layers=config.num_hidden_layers,
                                            num_attention_heads=config.num_attention_heads,
                                            intermediate_size=config.intermediate_size,
                                            intermediate_act_fn=get_activation(config.hidden_act),
                                            hidden_dropout_prob=config.hidden_dropout_prob,
                                            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                            initializer_range=config.initializer_range)
        self.pooler = tf.keras.layers.Dense(name="pooler/dense",
                                        units=config.hidden_size,
                                        activation=tf.tanh,
                                        kernel_initializer=create_initializer(config.initializer_range))

    #def call(self, features):
    def call(self, input_ids,
                    input_mask=None,
                    token_type_ids=None
        ):
        """Caller of BertModel.

        Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length].
        input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        
        Returns:
        All encoder outputs.
        
        input_ids      = features['input_ids']
        input_mask     = features.get('input_mask', None)
        token_type_ids = features.get('token_type_ids', None)
        """

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        # Perform embedding lookup on the word ids.
        self.embedding_output = self.embedding_lookup(inputs=input_ids)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = self.embedding_postprocessor(inputs=self.embedding_output,
                                                            token_type_ids=token_type_ids)

        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = self.transformer_model(inputs=self.embedding_output,
                                                        attention_mask=attention_mask,
                                                        do_return_all_layers=True)

        self.sequence_output = self.all_encoder_layers[-1]
        # The "pooler" converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size]. This is necessary for segment-level
        # (or segment-pair-level) classification tasks where we need a fixed
        # dimensional representation of the segment.

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        # self.pooled_output = self.pooler(first_token_tensor)
        self.pooled_output = {}
        return self.all_encoder_layers

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
        to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
        to the output of the embedding layer, after summing the word
        embeddings with the positional embeddings and the token type embeddings,
        then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_lookup.embedding_table

    def compute_loss(self, features):
        y_true = features["label_ids"]
        tf.keras.losses.SparseCategoricalCrossentropy()



def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    #cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.math.sqrt(2.0)))
    #return input_tensor * cdf
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (input_tensor + 0.044715 * tf.pow(input_tensor, 3)))))
    return input_tensor * cdf

def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def layer_norm(name=None):
    """Return layer normalization function."""
    # return tf.keras.layers.LayerNormalization(norm_axis=-1, params_axis=-1, name=name)
    return tf.keras.layers.LayerNormalization(axis=-1, name=name)

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.initializers.TruncatedNormal(stddev=initializer_range)


class embedding_lookup(tf.keras.layers.Layer):
    """Looks up words embeddings for id tensor."""
    def __init__(self,
                name,
                vocab_size,
                embedding_size=128,
                initializer_range=0.02,
                word_embedding_name="word_embeddings"):
        '''
        Constructor for embedding_lookup.
        
        Args:
        name: layer name.
        vocab_size: int. Size of the embedding vocabulary.
        embedding_size: int. Width of the word embeddings.
        initializer_range: float. Embedding initialization range.
        word_embedding_name: string. Name of the embedding table.
        '''
        super(embedding_lookup, self).__init__(name=name)    
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.word_embedding_name = word_embedding_name

    def build(self, input_shape):
        self.embedding_table = self.add_variable(name=self.word_embedding_name,
                                                shape=[self.vocab_size, self.embedding_size],
                                                initializer=create_initializer(self.initializer_range),
                                                dtype=tf.float32)  
        # This function assumes that the input is of shape [batch_size, seq_length,
        # num_inputs].
        #
        # If the input is a 2D tensor of shape [batch_size, seq_length], we
        # reshape to [batch_size, seq_length, 1].
    def call(self, inputs):
        '''
        Args:
        inputs: int32 Tensor of shape [batch_size, seq_length] containing word
            ids.

        Returns:
        float Tensor of shape [batch_size, seq_length, embedding_size].
        '''
    
        if inputs.shape.ndims == 2:
            inputs = tf.expand_dims(inputs, axis=[-1])
        output = tf.nn.embedding_lookup(self.embedding_table, inputs)
        input_shape = get_shape_list(inputs)
        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output

class embedding_postprocessor(tf.keras.layers.Layer):
    def __init__(self,
                name,
                use_token_type=False,
                token_type_vocab_size=16,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=0.02,
                max_position_embeddings=512,
                dropout_prob=0.1):
        """
        Constructor for embedding_postprocessor.
        
        Args:
        name: layer name.
        use_token_type: bool. Whether to add embeddings for `token_type_ids`.
        token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
        token_type_embedding_name: string. The name of the embedding table variable
            for token type ids.
        use_position_embeddings: bool. Whether to add position embeddings for the
            position of each token in the sequence.
        position_embedding_name: string. The name of the embedding table variable
            for positional embeddings.
        initializer_range: float. Range of the weight initialization.
        max_position_embeddings: int. Maximum sequence length that might ever be
            used with this model. This can be longer than the sequence length of
            input_tensor, but cannot be shorter.
        dropout_prob: float. Dropout probability applied to the final output tensor.
        """
        super(embedding_postprocessor, self).__init__(name=name)
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.token_type_embedding_name = token_type_embedding_name
        self.use_position_embeddings = use_position_embeddings
        self.position_embedding_name = position_embedding_name
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        
        self.dropout = tf.keras.layers.Dropout(rate=dropout_prob)
        self.layer_norm = layer_norm(name="LayerNorm")
        
    def build(self, input_shape):
        width = input_shape[2]
        if self.use_token_type:
            self.token_type_table = self.add_variable(name=self.token_type_embedding_name,
                                                        shape=[self.token_type_vocab_size, width],
                                                        initializer=create_initializer(self.initializer_range))
        if self.use_position_embeddings:
            self.full_position_embeddings = self.add_variable(name=self.position_embedding_name,
                                                                shape=[self.max_position_embeddings, width],
                                                                initializer=create_initializer(self.initializer_range))

    def call(self, inputs, token_type_ids):
        """Performs various post-processing on a word embedding tensor.
        
        Args:
        inputs: float Tensor of shape [batch_size, seq_length,
            embedding_size].
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            Must be specified if `use_token_type` is True.

        Returns:
        float tensor with same shape as `input`.

        Raises:
        ValueError: One of the tensor shapes or input values is invalid.
        """
        input_shape = get_shape_list(inputs, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        
        if seq_length > self.max_position_embeddings:
            raise ValueError("The seq length (%d) cannot be greater than "
                            "`max_position_embeddings` (%d)" %
                            (seq_length, self.max_position_embeddings))
        output = inputs
    
        if self.use_token_type:
            if token_type_ids is None:
                raise ValueError("`token_type_ids` must be specified if"
                                "`use_token_type` is True.")
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                            [batch_size, seq_length, width])
        output += token_type_embeddings
    
        if self.use_position_embeddings:
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            if seq_length < self.max_position_embeddings:
                position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                            [seq_length, -1])
            else:
                position_embeddings = self.full_position_embeddings

            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                            position_broadcast_shape)
            output += position_embeddings
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


class attention_layer(tf.keras.layers.Layer):
    '''Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.
    '''
    def __init__(self, 
                name, 
                num_attention_heads=1,
                size_per_head=512,
                query_act=None,
                key_act=None,
                value_act=None,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02):
        """Constructor of attention_layer

        Args:
        name: layer name
            num_attention_heads: int. Number of attention heads.
            size_per_head: int. Size of each attention head.
            query_act: (optional) Activation function for the query transform.
            key_act: (optional) Activation function for the key transform.
            value_act: (optional) Activation function for the value transform.
            attention_probs_dropout_prob: (optional) float. Dropout probability of the
                attention probabilities.
            initializer_range: float. Range of the weight initializer.
        """

        super(attention_layer, self).__init__(name=name)
        
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query_layer` = [B*F, N*H]
        self.query_layer = tf.keras.layers.Dense(num_attention_heads * size_per_head,
                                                activation=query_act,
                                                name="query",
                                                kernel_initializer=create_initializer(initializer_range))

        # `key_layer` = [B*T, N*H]
        self.key_layer = tf.keras.layers.Dense(num_attention_heads * size_per_head,
                                            activation=key_act,
                                            name="key",
                                            kernel_initializer=create_initializer(initializer_range))

        # `value_layer` = [B*T, N*H]
        self.value_layer = tf.keras.layers.Dense(num_attention_heads * size_per_head,
                                                activation=value_act,
                                                name="value",
                                                kernel_initializer=create_initializer(initializer_range))
        self.dropout = tf.keras.layers.Dropout(rate=attention_probs_dropout_prob)

    def transpose_for_scores(self, input_tensor, batch_size, num_attention_heads,
                            seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor  
  
    def call(self,
            inputs,
            from_tensor,
            to_tensor,
            attention_mask=None,
            do_return_2d_tensor=False,
            batch_size=None,
            from_seq_length=None,
            to_seq_length=None):
        """
        Args:
            inputs: default argument in 'call' function of keras layer.
            from_tensor: float Tensor of shape [batch_size, from_seq_length,
                from_width].
            to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
            attention_mask: (optional) int32 Tensor of shape [batch_size,
                from_seq_length, to_seq_length]. The values should be 1 or 0. The
                attention scores will effectively be set to -infinity for any positions in
                the mask that are 0, and will be unchanged for positions that are 1.
            do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
                * from_seq_length, num_attention_heads * size_per_head]. If False, the
                output will be of shape [batch_size, from_seq_length, num_attention_heads
                * size_per_head].
            batch_size: (Optional) int. If the input is 2D, this might be the batch size
                of the 3D version of the `from_tensor` and `to_tensor`.
            from_seq_length: (Optional) If the input is 2D, this might be the seq length
                of the 3D version of the `from_tensor`.
            to_seq_length: (Optional) If the input is 2D, this might be the seq length
                of the 3D version of the `to_tensor`.

        Returns:
            float Tensor of shape [batch_size, from_seq_length,
                num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
                true, this will be of shape [batch_size * from_seq_length,
                num_attention_heads * size_per_head]).

        Raises:
            ValueError: Any of the arguments or tensor shapes are invalid.
        """

        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
        
        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")
            
        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (batch_size is None or from_seq_length is None or to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)
        
        query_layer = self.query_layer(from_tensor_2d)
        key_layer = self.key_layer(to_tensor_2d)
        value_layer = self.value_layer(to_tensor_2d)
    
        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_layer, batch_size,
                                                self.num_attention_heads, from_seq_length,
                                                self.size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(key_layer, batch_size, self.num_attention_heads,
                                            to_seq_length, self.size_per_head)
        
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                    1.0 / math.sqrt(float(self.size_per_head)))
        
        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder
            
        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*V]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N*V]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer


class transformer_model(tf.keras.Model):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """
    def __init__(self, 
                name,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                intermediate_act_fn=gelu,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02):
        """Constructor of transformer_model

        Args:
        name: model name
        hidden_size: int. Hidden size of the Transformer.
        num_hidden_layers: int. Number of layers (blocks) in the Transformer.
        num_attention_heads: int. Number of attention heads in the Transformer.
        intermediate_size: int. The size of the "intermediate" (a.k.a., feed
            forward) layer.
        intermediate_act_fn: function. The non-linear activation function to apply
            to the output of the intermediate/feed-forward layer.
        hidden_dropout_prob: float. Dropout probability for the hidden layers.
        attention_probs_dropout_prob: float. Dropout probability of the attention
            probabilities.
        initializer_range: float. Range of the initializer (stddev of truncated
            normal).
        """
        super(transformer_model, self).__init__(name=name)
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        attention_head_size = int(hidden_size / num_attention_heads)
        
        self.attention_heads = []
        self.attention_outputs = []
        self.attention_layer_norms = []
        self.intermediate_outputs = []
        self.layer_outputs = []
        self.output_layer_norms = []
        for layer_idx in range(num_hidden_layers):
            attention_head = attention_layer(
                        name="layer_%d" % layer_idx + "/attention" + "/self", 
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range)
            self.attention_heads.append(attention_head)
            
            attention_output = tf.keras.layers.Dense(                
                        hidden_size,
                        name="layer_%d" % layer_idx + "/attention" + "/output" + "/dense",
                        kernel_initializer=create_initializer(initializer_range))
            self.attention_outputs.append(attention_output)
            
            attention_layer_norm = layer_norm(name="layer_%d" % layer_idx + "/attention/output/LayerNorm")
            self.attention_layer_norms.append(attention_layer_norm)

            intermediate_output = tf.keras.layers.Dense(
                    intermediate_size,
                    name="layer_%d" % layer_idx + "/intermediate" + "/dense",
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))
            self.intermediate_outputs.append(intermediate_output)

            layer_output = tf.keras.layers.Dense(
                    hidden_size,
                    name="layer_%d" % layer_idx + "/output" + "/dense",
                    kernel_initializer=create_initializer(initializer_range))
            self.layer_outputs.append(layer_output)
            
            output_layer_norm = layer_norm(name="layer_%d" % layer_idx + "/output/LayerNorm")
            self.output_layer_norms.append(output_layer_norm)
        
        self.dropout = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
  
    def call(self,
            inputs,
            attention_mask=None,
            do_return_all_layers=False):
        '''
        Args:
        inputs: float Tensor of shape [batch_size, seq_length, hidden_size].
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
            seq_length], with 1 for positions that can be attended to and 0 in
            positions that should not be.
        
        do_return_all_layers: Whether to also return all layers or just the final
            layer.

        Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size], the final
        hidden layer of the Transformer.

        Raises:
        ValueError: A Tensor shape or parameter is invalid.
        '''
        input_shape = get_shape_list(inputs, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]
    
        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                            (input_width, self.hidden_size))
        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = reshape_to_matrix(inputs)
    
        all_layer_outputs = []
        for layer_idx in range(self.num_hidden_layers):
            layer_input = prev_output
            
            attention_heads = []
            attention_head = self.attention_heads[layer_idx](inputs=None,
                                                            from_tensor=layer_input,
                                                            to_tensor=layer_input,
                                                            attention_mask=attention_mask,
                                                            do_return_2d_tensor=True,
                                                            batch_size=batch_size,
                                                            from_seq_length=seq_length,
                                                            to_seq_length=seq_length)
            attention_heads.append(attention_head)
                
            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection.
                attention_output = tf.concat(attention_heads, axis=-1)
            
            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            attention_output = self.attention_outputs[layer_idx](attention_output)
            attention_output = self.dropout(attention_output)
            attention_output = self.attention_layer_norms[layer_idx](attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            intermediate_output = self.intermediate_outputs[layer_idx](attention_output)

            # Down-project back to `hidden_size` then add the residual.
            layer_output = self.layer_outputs[layer_idx](intermediate_output)
            layer_output = self.dropout(layer_output)
            layer_output = self.output_layer_norms[layer_idx](layer_output + attention_output)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)
        
        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def get_shape_list(tensor, expected_rank=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    
    # Cannot convert this function to autograph.
    if expected_rank is not None:
        assert_rank(tensor, expected_rank)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                        (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "the actual rank `%d` (shape = %s) is not equal to the expected rank `%s`" %
            (actual_rank, str(tensor.shape), str(expected_rank)))
