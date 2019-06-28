import tensorflow as tf
import tensorflow_probability as tfp

features=[], labels=[] #assuming features and labels

#for completeness: including tensorflow probablity models

model = tf.keras.Sequential([
    tf.keras.layers.Reshape([32, 32, 3]),
    tfp.layers.Convolution2DReparameterization(
        64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                 strides=[2, 2],
                                 padding='SAME'),
    tf.keras.layers.Flatten(),
    tfp.layers.DenseReparameterization(10),
])

logits = model(features)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
kl = sum(model.losses)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)

tfp.layers.DenseReparameterization.losses