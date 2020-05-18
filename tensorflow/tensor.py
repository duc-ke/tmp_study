import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_train = [1,2,3]
y_train = [1,2,3]
W = tf.Variable(tf.random.normal([1]), name = 'weights')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

# our hypothesis 
hypothesis = x_train * W + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#Gradient Descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range (2001):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b)) 

