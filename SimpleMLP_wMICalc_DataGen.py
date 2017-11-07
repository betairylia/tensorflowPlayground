import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Global config variables
dataCount = 1000
numEpochs = 1
batch_size = 100
learning_rate = 0.1

#Network creation
numInputs = 1
numOutputs = 2
layerStructure = [numInputs, 8, 6, numOutputs] #Input -> hidden -> ... -> Output

def gen_data(size=1000):
    X = np.array(np.random.choice(2, size=(size,numInputs)))
    Y = []
    for i in range(size):
        if X[i] == 1:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_partition_length, batch_size, numInputs], dtype=np.float32)
    data_y = np.zeros([batch_partition_length, batch_size, numOutputs], dtype=np.float32)
    for i in range(batch_partition_length):
        data_x[i] = raw_x[batch_size * i:batch_size * (i + 1)]
        data_y[i] = raw_y[batch_size * i:batch_size * (i + 1)]

    for i in range(batch_partition_length):
        x = data_x[i]
        y = data_y[i]
        yield (x, y)

def gen_epochs(n):
    for i in range(n):
        yield gen_batch(gen_data(dataCount), batch_size)

def calculateMI_Plane(x, t, y):
    # TODO

#Variable used for calculation
weight = []
bias = []
layer = []
depth = len(layerStructure)

X = tf.placeholder("float", [None, layerStructure[0]])
Y = tf.placeholder("float", [None, layerStructure[depth - 1]])

def initNetwork():
    for i in range(depth - 1):
        weight.append(tf.Variable(tf.random_normal([layerStructure[i], layerStructure[i + 1]])))
        bias.append(tf.Variable(tf.random_normal([layerStructure[i + 1]])))

def multilayer_perceptron(x):
    #Input layer
    layer.append(
        tf.tanh(tf.add(tf.matmul(x, weight[0]), bias[0])))
    #Hidden layers
    for i in range(1, depth - 2):
        layer.append(
            tf.tanh(tf.add(tf.matmul(layer[i - 1], weight[i]), bias[i])))
    #Output softmax
    layer.append(
        tf.add(tf.matmul(layer[depth - 3], weight[depth - 2]), bias[depth - 2]))

    return layer[depth - 2]

initNetwork()
logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits = logits, labels = Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

def train_network(num_epochs, verbose=True, printInterval=1, calculateMI=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs)):
            training_loss = 0
            if verbose:
                print("\nEPOCH", idx)
            for step, (Xdata, Ydata) in enumerate(epoch):
                _, c = \
                    sess.run([train_op,
                              loss_op],
                                  feed_dict={X:Xdata, Y:Ydata})
                training_loss += c

                #calculate Mutual Information
                if calculateMI:


                #show training status
                if step % printInterval == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last ", printInterval, " step(s):", training_loss/printInterval)
                    training_losses.append(training_loss/printInterval)
                    training_loss = 0
    return training_losses

training_losses = train_network(numEpochs)
plt.plot(training_losses)
plt.show()
