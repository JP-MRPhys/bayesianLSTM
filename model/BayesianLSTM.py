import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, LSTMCell
from model.utils import variationalPosterior


class BayesianLSTMCell(LSTMCell):

    def __init__(self, num_units, prior, is_training, name, **kwargs):

        super(BayesianLSTMCell, self).__init__(num_units, **kwargs)

        self.w = None
        self.b = None
        self.prior = prior
        self.layer_name = name
        self.isTraining = is_training
        self.num_units = num_units
        self.kl_loss=None

        print("Creating lstm layer:" + name)


    def call(self, inputs, state):

        if self.w is None:

            size = inputs.get_shape()[-1].value
            self.w, self.w_mean, self.w_sd = variationalPosterior((size+self.num_units, 4*self.num_units), self.layer_name+'_weights', self.prior, self.isTraining)
            self.b, self.b_mean, self.b_sd = variationalPosterior((4*self.num_units,1), self.layer_name+'_bias', self.prior, self.isTraining)

        cell, hidden = state
        concat_inputs_hidden = tf.concat([inputs, hidden], 1)
        concat_inputs_hidden = tf.nn.bias_add(tf.matmul(concat_inputs_hidden, self.w), tf.squeeze(self.b))
        # Gates: Input, New, Forget and Output
        i, j, f, o = tf.split(value=concat_inputs_hidden, num_or_size_splits=4, axis=1)
        new_cell = (cell * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
        new_hidden = self._activation(new_cell) * tf.sigmoid(o)
        new_state = LSTMStateTuple(new_cell, new_hidden)

        return new_hidden, new_state






