import tensorflow as tf
from utils import *


class DiscriminatorWithStyle(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 rnn_depth, rnn_units, num_tokens, num_meta_features, num_ref_features):
        super(DiscriminatorWithStyle, self).__init__()

        self.rnn_depth = rnn_depth
        self.rnn_units = rnn_units

        self.embedding = [tf.keras.layers.Dense(
            units, use_bias=False, kernel_initializer=create_linear_initializer(n)) for units, n in
            zip(emb_units, num_tokens)]

        self.embedding_dropout = [tf.keras.layers.Dropout(rate) for rate in emb_dropout_rate]

        self.projection = tf.keras.layers.Dense(
            proj_units, activation='relu',
            kernel_initializer=create_linear_initializer(sum(emb_units) + num_meta_features + sum(num_ref_features)))

        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

        self.lstm = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(rnn_units) for _ in range(rnn_depth)]),
            return_sequences=True,
            return_state=True)

        self.outputs = tf.keras.layers.Dense(1)

    def call(self, inputs, memory, reference, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES],
                      ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for lstm layer
        """
        m, (p, d, r) = inputs
        style_p, style_d, style_r = reference

        p = self.embedding[0](p)
        d = self.embedding[1](d)
        r = self.embedding[2](r)

        p = self.embedding_dropout[0](p, training=training)
        d = self.embedding_dropout[1](d, training=training)
        r = self.embedding_dropout[2](r, training=training)

        x = tf.concat([p, d, r, m, style_p, style_d, style_r], axis=-1)  # [pitch, duration, rest ; meta ; style]

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        x, *memory = self.lstm(tf.expand_dims(x, 1), initial_state=memory)

        x = self.outputs(tf.squeeze(x, 1))

        return x, memory

    def initial_state(self, batch_size):
        """
        This method returns initial state of lstm layers.
        """
        return [[tf.zeros((batch_size, self.rnn_units)),
                 tf.zeros((batch_size, self.rnn_units))] for _ in range(self.rnn_depth)]


"""## Discriminator"""

class Discriminator(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
             rnn_depth, rnn_units, num_tokens, num_meta_features):
        super(Discriminator, self).__init__()

        self.rnn_depth = rnn_depth
        self.rnn_units = rnn_units

        self.embedding = [tf.keras.layers.Dense(
          units, use_bias=False, kernel_initializer=create_linear_initializer(n)) for units, n in zip(emb_units, num_tokens)]

        self.embedding_dropout = [tf.keras.layers.Dropout(rate) for rate in emb_dropout_rate]
          
        self.projection = tf.keras.layers.Dense(
          proj_units, activation='relu', kernel_initializer=create_linear_initializer(sum(emb_units)+num_meta_features))

        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

        self.lstm = tf.keras.layers.RNN(
        tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(rnn_units) for _ in range(rnn_depth)]),
        return_sequences=True,
        return_state=True)

        self.outputs = tf.keras.layers.Dense(1)

    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES],
                      ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for lstm layer
        """
        m, (p, d, r) = inputs

        p = self.embedding[0](p)
        d = self.embedding[1](d)
        r = self.embedding[2](r)

        p = self.embedding_dropout[0](p, training=training)
        d = self.embedding_dropout[1](d, training=training)
        r = self.embedding_dropout[2](r, training=training)

        x = tf.concat([p, d, r, m], axis=-1) # [pitch, duration, rest ; meta]

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        x, *memory = self.lstm(tf.expand_dims(x, 1), initial_state=memory)

        x = self.outputs(tf.squeeze(x, 1))

        return x, memory

    def initial_state(self, batch_size):
        """
        This method returns initial state of lstm layers.
        """
        return [[tf.zeros((batch_size, self.rnn_units)),
                tf.zeros((batch_size, self.rnn_units))] for _ in range(self.rnn_depth)]
