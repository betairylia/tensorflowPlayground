import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mutual_info_score
from termcolor import colored, cprint

#Global config variables
dataCount = 300000
numEpochs = 1
batch_size = 150
learning_rate = 0.1

MI_Interval = 40
num_loop = 20

#Network creation
numInputs = 12
numOutputs = 4
layerStructure = [numInputs, 12, 10, 6, 10, 6, numOutputs] #Input -> hidden -> ... -> Output

def get_Y(input, inputLenth, classCount):
    stepLenth = inputLenth // classCount
    targetClass = 0
    maxArrayNum = 0
    for i in range(classCount):
        n = array_to_number(input[i * stepLenth : (i + 1) * stepLenth])
        if(maxArrayNum <= n):
            maxArrayNum = n
            targetClass = i

    result = []
    for i in range(classCount):
        if i == targetClass:
            result.append(1)
        else:
            result.append(0)

    return result

def gen_data(size=1000):
    X = np.array(np.random.choice(2, size=(size,numInputs)))
    Y = []
    for i in range(size):
        Y.append(get_Y(X[i], numInputs, numOutputs))
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

def array_to_number(a):
    result = 0
    for i in range(len(a)):
        result *= 2
        if(a[len(a) - i - 1] > 0):
            result += 1
    return result

def calculate_prob(a):
    dict_Index = dict()
    count = np.zeros(a.size)

    for i in range(a.size):
        if a[i] not in dict_Index.keys():
            dict_Index[a[i]] = len(dict_Index)
        count[dict_Index[a[i]]] += 1

    prob = np.zeros(len(dict_Index))

    for i in range(len(dict_Index)):
        prob[i] = count[i] / a.size

    return prob

def calculate_jointProb(a, b):
    dict_IndexA = dict()

    for i in range(a.size):
        if a[i] not in dict_IndexA.keys():
            dict_IndexA[a[i]] = len(dict_IndexA)

    ####

    dict_IndexB = dict()

    for i in range(b.size):
        if b[i] not in dict_IndexB.keys():
            dict_IndexB[b[i]] = len(dict_IndexB)

    prob = np.zeros((len(dict_IndexA), len(dict_IndexB)))

    for i in range(a.size):
        for j in range(b.size):
            prob[dict_IndexA[a[i]], dict_IndexB[b[j]]] += 1
    prob /= (a.size * b.size)

    return prob

def calculateMI_Plane(x, t, y, binCount = 24):
    # Push T into bins
    # T_bins = np.zeros((len(t), binCount))
    T_Nums = np.zeros(len(t))
    X_Nums = np.zeros(len(x))
    Y_Nums = np.zeros(len(y))

    for i in range(len(t)):
        # T_bins[i] = np.histogram(t[i], bins=binCount, range=(-1.0, 1.0))[0] # throw out bin edges
        T_Nums[i] = array_to_number(np.histogram(t[i], bins=binCount, range=(-1.0, 1.0))[0])
        X_Nums[i] = array_to_number(x[i])
        Y_Nums[i] = array_to_number(y[i])
        # cprint("- X: %d" % X_Nums[i], 'green')
        # cprint("- T: %d" % T_Nums[i], 'yellow')
        # cprint("- Y: %d" % Y_Nums[i], 'red')

    # probXT = calculate_jointProb(X_Nums, T_Nums)
    # probTY = calculate_jointProb(T_Nums, Y_Nums)

    # cprint(probXT, 'yellow')
    # cprint(probTY, 'cyan')

    # return (mutual_info_score(None, None, contingency=probXT), mutual_info_score(None, None, contingency=probTY))
    return (mutual_info_score(X_Nums, T_Nums), mutual_info_score(T_Nums, Y_Nums))

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

def train_network(num_epochs, verbose=True, printInterval=100, calculateMI=True, num_loop=1, MI_Interval=20):
    with tf.Session() as sess:

        MI_Plane_XT = []
        MI_Plane_TY = []
        if calculateMI:
            for i in range(len(layer) - 1):
                MI_Plane_XT.append([])
                MI_Plane_TY.append([])
                for j in range(dataCount // batch_size // MI_Interval):
                    MI_Plane_XT[i].append(0)
                    MI_Plane_TY[i].append(0)

        for loopCount in range(num_loop):
            sess.run(tf.global_variables_initializer())
            training_losses = []

            if verbose:
                print(colored("\n*****\nLOOP %d Begins\n*****" % (loopCount + 1), 'cyan', attrs=['bold']))

            for idx, epoch in enumerate(gen_epochs(num_epochs)):
                training_loss = 0
                if verbose:
                    cprint("EPOCH %d" % idx, 'magenta')
                for step, (Xdata, Ydata) in enumerate(epoch):
                    _, c = \
                        sess.run([train_op,
                                  loss_op],
                                      feed_dict={X:Xdata, Y:Ydata})
                    training_loss += c

                    #calculate Mutual Information
                    if calculateMI and step % MI_Interval == 0:
                        layerData = \
                            sess.run(layer, feed_dict={X:Xdata, Y:Ydata})
                        # print (layerData[0])

                        for layerId in range(len(layerData) - 1):
                            (IX_T, IT_Y) = calculateMI_Plane(Xdata, layerData[layerId], Ydata)
                            MI_Plane_XT[layerId][step // MI_Interval] += IX_T
                            MI_Plane_TY[layerId][step // MI_Interval] += IT_Y

                            if step % printInterval == 0:
                                print(colored("MI Plane position of Layer #%d: " % (layerId + 1), 'green') + colored("X: %lf, Y: %lf" % (IX_T, IT_Y), 'blue'))

                    #show training status
                    if step % printInterval == 0 and step > 0:
                        if verbose:
                            print(colored("Average loss at step %d for last %d step(s): " % (printInterval, step), 'yellow') + colored("%lf" % (training_loss/printInterval), 'red'))
                        training_losses.append(training_loss/printInterval)
                        training_loss = 0

        if calculateMI:
            for i in range(len(layer) - 1):
                for j in range(dataCount // batch_size // MI_Interval):
                    MI_Plane_XT[i][j] /= num_loop
                    MI_Plane_TY[i][j] /= num_loop

    return (training_losses, MI_Plane_XT, MI_Plane_TY)

(training_losses, miX, miY) = train_network(numEpochs, num_loop = num_loop, MI_Interval = MI_Interval)
plt.plot(training_losses)

plt.show()
plt.clf()
plt.cla()
plt.close()
plt.close('all')

#draw the MI plot

colorLayer = []
totalStepsMI = dataCount // batch_size // MI_Interval
colorMaps = [cm.Greys, cm.Blues, cm.YlOrBr, cm.RdPu, cm.YlGn]
for i in range(len(layer) - 1):
    colorLayer.append([])
    for j in range(totalStepsMI):
        colorLayer[i].append(colorMaps[i % len(colorMaps)](j / totalStepsMI))

for i in range(len(layer) - 1):
    plt.scatter(miX[i], miY[i], c=colorLayer[i], zorder=2)
    plt.plot(miX[i], miY[i], c=[0.3, 0.3, 0.3], ls='-', zorder=1)
plt.show()
