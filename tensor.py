import tensorflow as tf

# X and Y data
# dfadfasdfadsfdsfas
x_train = [1,2,3]
y_train = [1,2,3]
W = tf.Variable(tf.random.normal([1]), name = 'weights')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

# our hypothesis 
hypothesis = x_train * W + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
