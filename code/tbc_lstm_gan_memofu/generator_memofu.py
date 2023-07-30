"""
@Project: tbc_lstm_gan_style
@File: generator_memofu.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/09/19
"""

import tensorflow as tf
from utils import *


class GeneratorWithMemoryFusion(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 rnn_depth, rnn_units, num_tokens, num_meta_features, fusion_layer=0):
        super(GeneratorWithMemoryFusion, self).__init__()

        self.p_subg = SubGenerator(emb_units[0], proj_units[0], emb_dropout_rate[0],
                                   proj_dropout_rate[0], rnn_depth[0], rnn_units[0],
                                   num_tokens[0], num_meta_features)

        self.d_subg = SubGenerator(emb_units[1], proj_units[1], emb_dropout_rate[1],
                                   proj_dropout_rate[1], rnn_depth[1], rnn_units[1],
                                   num_tokens[1], num_meta_features)

        self.r_subg = SubGenerator(emb_units[2], proj_units[2], emb_dropout_rate[2],
                                   proj_dropout_rate[2], rnn_depth[2], rnn_units[2],
                                   num_tokens[2], num_meta_features)

        self.proj_d2p = tf.keras.layers.Dense(rnn_units[0], activation='relu')
        self.proj_r2p = tf.keras.layers.Dense(rnn_units[0], activation='relu')

        self.proj_p2d = tf.keras.layers.Dense(rnn_units[1], activation='relu')
        self.proj_r2d = tf.keras.layers.Dense(rnn_units[1], activation='relu')

        self.proj_p2r = tf.keras.layers.Dense(rnn_units[2], activation='relu')
        self.proj_d2r = tf.keras.layers.Dense(rnn_units[2], activation='relu')

        self.fusion_layer = fusion_layer

    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES],
                      ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for lstm for each sub generator
        :type  memory: tuple
        """

        m, (p, d, r) = inputs
        memory_p, memory_d, memory_r = memory

        l = self.fusion_layer
        # memory[][1]: memory
        memory_p[l][1] = memory_p[l][1] + self.proj_d2p(memory_d[l][1]) + self.proj_r2p(memory_r[l][1])
        memory_d[l][1] = memory_d[l][1] + self.proj_p2d(memory_p[l][1]) + self.proj_r2d(memory_r[l][1])
        memory_r[l][1] = memory_r[l][1] + self.proj_p2r(memory_p[l][1]) + self.proj_d2r(memory_d[l][1])

        p, memory_p = self.p_subg((m, p), memory_p, training=training)
        d, memory_d = self.d_subg((m, d), memory_d, training=training)
        r, memory_r = self.r_subg((m, r), memory_r, training=training)

        return (p, d, r), (memory_p, memory_d, memory_r)


"""### SubGenerator"""

class SubGenerator(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 rnn_depth, rnn_units, num_tokens, num_meta_features):
        super(SubGenerator, self).__init__()

        self.rnn_depth = rnn_depth
        self.rnn_units = rnn_units

        self.embedding = tf.keras.layers.Dense(
          emb_units, use_bias=False, kernel_initializer=create_linear_initializer(num_tokens))

        self.embedding_dropout = tf.keras.layers.Dropout(emb_dropout_rate)

        self.projection = tf.keras.layers.Dense(
          proj_units, activation='relu', kernel_initializer=create_linear_initializer(emb_units+num_meta_features))

        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

        self.lstm = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(rnn_units) for _ in range(rnn_depth)]),
            return_sequences=True,
            return_state=True)

        self.outputs = tf.keras.layers.Dense(
          num_tokens, kernel_initializer=create_linear_initializer(rnn_units))

    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta (i.e. syllable) and note attribute (pitch, duration or rest)
        :type. inputs: tuple
        :shape inputs: ([None, NUM_META_FEATURES], [None, NUM_[.]_TOKENS])

        :param memory: lstm memory
        """
        m, n = inputs

        n = self.embedding(n)
        n = self.embedding_dropout(n, training=training)

        x = tf.concat([n, m], axis=-1) # [p+d+r+m]

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        x, *memory = self.lstm(tf.expand_dims(x, 1), initial_state=memory)
        # memory[][0]: output sequences
        # memory[-1][0] == x
        # memory[][1]: memory

        x = self.outputs(tf.squeeze(x, 1))

        return x, memory

    def initial_state(self, batch_size):
        """
        This method returns initial state of lstm layers.
        """
        return [[tf.zeros((batch_size, self.rnn_units)), tf.zeros((batch_size, self.rnn_units))] for _ in range(self.rnn_depth)]


class GeneratorWithMemoryFusionAndStyle(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 rnn_depth, rnn_units, num_tokens, num_meta_features, num_ref_features, fusion_layer=0):
        super(GeneratorWithMemoryFusionAndStyle, self).__init__()

        self.p_subg = SubGeneratorWithMemoryFusionAndStyle(emb_units[0], proj_units[0], emb_dropout_rate[0],
                                                           proj_dropout_rate[0], rnn_depth[0], rnn_units[0],
                                                           num_tokens[0], num_meta_features, num_ref_features[0])

        self.d_subg = SubGeneratorWithMemoryFusionAndStyle(emb_units[1], proj_units[1], emb_dropout_rate[1],
                                                           proj_dropout_rate[1], rnn_depth[1], rnn_units[1],
                                                           num_tokens[1], num_meta_features, num_ref_features[1])

        self.r_subg = SubGeneratorWithMemoryFusionAndStyle(emb_units[2], proj_units[2], emb_dropout_rate[2],
                                                           proj_dropout_rate[2], rnn_depth[2], rnn_units[2],
                                                           num_tokens[2], num_meta_features, num_ref_features[2])

        self.proj_d2p = tf.keras.layers.Dense(rnn_units[0], activation='relu')
        self.proj_r2p = tf.keras.layers.Dense(rnn_units[0], activation='relu')

        self.proj_p2d = tf.keras.layers.Dense(rnn_units[1], activation='relu')
        self.proj_r2d = tf.keras.layers.Dense(rnn_units[1], activation='relu')

        self.proj_p2r = tf.keras.layers.Dense(rnn_units[2], activation='relu')
        self.proj_d2r = tf.keras.layers.Dense(rnn_units[2], activation='relu')

        self.fusion_layer = fusion_layer

    def call(self, inputs, memory, reference, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES],
                      ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for lstm for each sub generator
        :type  memory: tuple
        """

        m, (p, d, r) = inputs
        style_p, style_d, style_r = reference
        memory_p, memory_d, memory_r = memory

        l = self.fusion_layer
        # memory[][1]: memory
        memory_p[l][1] = memory_p[l][1] + self.proj_d2p(memory_d[l][1]) + self.proj_r2p(memory_r[l][1])
        memory_d[l][1] = memory_d[l][1] + self.proj_p2d(memory_p[l][1]) + self.proj_r2d(memory_r[l][1])
        memory_r[l][1] = memory_r[l][1] + self.proj_p2r(memory_p[l][1]) + self.proj_d2r(memory_d[l][1])

        p, memory_p = self.p_subg((m, p), memory_p, style_p, training=training)
        d, memory_d = self.d_subg((m, d), memory_d, style_d, training=training)
        r, memory_r = self.r_subg((m, r), memory_r, style_r, training=training)

        return (p, d, r), (memory_p, memory_d, memory_r)


"""### SubGenerator"""

class SubGeneratorWithMemoryFusionAndStyle(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 rnn_depth, rnn_units, num_tokens, num_meta_features, num_ref_features):
        super(SubGeneratorWithMemoryFusionAndStyle, self).__init__()

        self.rnn_depth = rnn_depth
        self.rnn_units = rnn_units

        self.embedding = tf.keras.layers.Dense(
            emb_units, use_bias=False, kernel_initializer=create_linear_initializer(num_tokens))

        self.embedding_dropout = tf.keras.layers.Dropout(emb_dropout_rate)

        self.projection = tf.keras.layers.Dense(
            proj_units, activation='relu',
            kernel_initializer=create_linear_initializer(emb_units + num_meta_features + num_ref_features))

        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

        self.lstm = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(rnn_units) for _ in range(rnn_depth)]),
            return_sequences=True,
            return_state=True)

        self.outputs = tf.keras.layers.Dense(
          num_tokens, kernel_initializer=create_linear_initializer(rnn_units))

    def call(self, inputs, memory, reference, training=False):
        """
        :param inputs: meta (i.e. syllable) and note attribute (pitch, duration or rest)
        :type. inputs: tuple
        :shape inputs: ([None, NUM_META_FEATURES], [None, NUM_[.]_TOKENS])

        :param memory: lstm memory
        """
        m, n = inputs

        n = self.embedding(n)
        n = self.embedding_dropout(n, training=training)

        # print(m)
        # print(n)
        # print(r)
        # x = tf.concat([n, m, r], axis=-1)  # [p+d+r+m]
        # tf.cast(x, tf.float64)
        # print(x)

        x = tf.concat([n, m, reference], axis=-1)  # p + d + r + meta + style
        # x = tf.concat([x, reference], axis=-1)

        # print(x)

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        x, *memory = self.lstm(tf.expand_dims(x, 1), initial_state=memory)

        x = self.outputs(tf.squeeze(x, 1))

        return x, memory

    def initial_state(self, batch_size):
        """
        This method returns initial state of lstm layers.
        """
        return [[tf.zeros((batch_size, self.rnn_units)), tf.zeros((batch_size, self.rnn_units))] for _ in range(self.rnn_depth)]

