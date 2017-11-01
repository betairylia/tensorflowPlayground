#get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("Data/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# that should be something like gaussian instead of zeros
with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    label = tf.placeholder(tf.float32, [None, 10], name="input_label")

with tf.variable_scope("Layer-1"):
    W = tf.Variable(tf.random_normal([784, 64]), name="W")
    b = tf.Variable(tf.random_normal([64]), name="b")
    y1 = tf.matmul(x, W) + b

    variable_summaries(W)

with tf.variable_scope("Layer-2"):
    W2 = tf.Variable(tf.random_normal([64, 10]), name="W")
    b2 = tf.Variable(tf.random_normal([10]), name="b")
    y = tf.matmul(y1,W2) + b2

with tf.variable_scope("loss"):
    loss = tf.reduce_mean(tf.square(y-label))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=label))
# loss = tf.reduce_mean(tf.square(y-label))

train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

train_writer = tf.summary.FileWriter("Logs", sess.graph)

sess.run(tf.global_variables_initializer())

for itr in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))

        merged = tf.summary.merge_all()
        summary = sess.run(merged, feed_dict={x:batch_xs,label:batch_ys})
        train_writer.add_summary(summary, itr)
