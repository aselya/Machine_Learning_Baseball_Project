import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


print("inports worked")

from sklearn.preprocessing import normalize

csv_Dataframe = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv")
csv_Dataframe = csv_Dataframe.fillna(1) #replaces all NAN with 1 representing average
#one hot encoding
dataframe_lb = LabelBinarizer()
Y = dataframe_lb.fit_transform(csv_Dataframe.Classification.values)
print("Y" + str(Y))


FEATURES = csv_Dataframe.columns[0:26]
print("features: "+ str(FEATURES))
X_data = csv_Dataframe[FEATURES].as_matrix()
X_data = normalize(X_data)

#print(X_data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=1)
X_train.shape
print("X_train shape" + str(X_train.shape)) #1820, 26| features
print("Y_train shape" + str(y_train.shape)) #1820, 4 | 4 -> bad tema, average team, good team, playoff team



# Parameters
learning_rate = 0.01
training_epochs = 1000


n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 1st layer number of neurons

n_input = X_train.shape[1] # 26
n_classes = y_train.shape[1] # 4


# Inputs
X = tf.placeholder("float", shape=[None, n_input])
y = tf.placeholder("float", shape=[None, n_classes])

# Dictionary of Weights and Biases
weights = {
  'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
  'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
  'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
  'b1': tf.Variable(tf.random_normal([n_hidden_1])),
  'b2': tf.Variable(tf.random_normal([n_hidden_2])),
  'out': tf.Variable(tf.random_normal([n_classes]))
}

def forward_propagation(x):
    # Hidden layer1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

yhat = forward_propagation(X)
ypredict = tf.argmax(yhat, axis=1)


# Backward propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()

from datetime import datetime
startTime = datetime.now()

with tf.Session() as sess:
    sess.run(init)

    #writer.add_graph(sess.graph)
    #EPOCHS
    for epoch in range(training_epochs):
        #Stochasting Gradient Descent
        for i in range(len(X_train)):
            summary = sess.run(train_op, feed_dict={X: X_train[i: i + 1], y: y_train[i: i + 1]})

        train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredict, feed_dict={X: X_train, y: y_train}))
        test_accuracy  = np.mean(np.argmax(y_test, axis=1) == sess.run(ypredict, feed_dict={X: X_test, y: y_test}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
        #print("Epoch = %d, train accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy))

    sess.close()
print("Time taken:", datetime.now() - startTime)

'''
results


inports worked
Y[[0 1 0 0]
 [0 0 0 1]
 [0 0 1 0]
 ...
 [1 0 0 0]
 [0 1 0 0]
 [0 1 0 0]]
features: Index(['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF',
       'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA',
       'SOA', 'E', 'DP', 'FP'],
      dtype='object')
/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/NN_nov28_3.py:23: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
  X_data = csv_Dataframe[FEATURES].as_matrix()
X_train shape(1820, 26)
Y_train shape(1820, 4)
WARNING:tensorflow:From /Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/NN_nov28_3.py:82: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See `tf.nn.softmax_cross_entropy_with_logits_v2`.

2018-11-28 12:54:10.253828: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Epoch = 1, train accuracy = 42.91%, test accuracy = 42.18%
Epoch = 2, train accuracy = 42.20%, test accuracy = 40.51%
Epoch = 3, train accuracy = 45.60%, test accuracy = 44.36%
Epoch = 4, train accuracy = 54.45%, test accuracy = 53.72%
Epoch = 5, train accuracy = 49.40%, test accuracy = 48.21%
Epoch = 6, train accuracy = 50.49%, test accuracy = 49.74%
Epoch = 7, train accuracy = 50.60%, test accuracy = 50.13%
Epoch = 8, train accuracy = 53.24%, test accuracy = 51.92%
Epoch = 9, train accuracy = 59.78%, test accuracy = 58.08%
Epoch = 10, train accuracy = 61.87%, test accuracy = 59.62%
Epoch = 11, train accuracy = 61.92%, test accuracy = 61.15%
Epoch = 12, train accuracy = 63.02%, test accuracy = 61.79%
Epoch = 13, train accuracy = 61.21%, test accuracy = 60.26%
Epoch = 14, train accuracy = 56.43%, test accuracy = 54.87%
Epoch = 15, train accuracy = 60.88%, test accuracy = 60.26%
Epoch = 16, train accuracy = 61.87%, test accuracy = 61.79%
Epoch = 17, train accuracy = 62.03%, test accuracy = 62.05%
Epoch = 18, train accuracy = 61.48%, test accuracy = 61.79%
Epoch = 19, train accuracy = 61.59%, test accuracy = 61.54%
Epoch = 20, train accuracy = 62.53%, test accuracy = 63.21%
Epoch = 21, train accuracy = 62.14%, test accuracy = 62.05%
Epoch = 22, train accuracy = 57.31%, test accuracy = 56.79%
Epoch = 23, train accuracy = 62.80%, test accuracy = 62.82%
Epoch = 24, train accuracy = 64.56%, test accuracy = 63.85%
Epoch = 25, train accuracy = 60.22%, test accuracy = 59.87%
Epoch = 26, train accuracy = 61.54%, test accuracy = 61.92%
Epoch = 27, train accuracy = 61.54%, test accuracy = 63.21%
Epoch = 28, train accuracy = 62.09%, test accuracy = 63.46%
Epoch = 29, train accuracy = 63.19%, test accuracy = 63.08%
Epoch = 30, train accuracy = 62.36%, test accuracy = 63.21%
Epoch = 31, train accuracy = 62.25%, test accuracy = 62.44%
Epoch = 32, train accuracy = 62.09%, test accuracy = 61.67%
Epoch = 33, train accuracy = 62.58%, test accuracy = 62.05%
Epoch = 34, train accuracy = 62.36%, test accuracy = 62.18%
Epoch = 35, train accuracy = 61.76%, test accuracy = 62.56%
Epoch = 36, train accuracy = 63.90%, test accuracy = 63.33%
Epoch = 37, train accuracy = 62.69%, test accuracy = 63.33%
Epoch = 38, train accuracy = 63.24%, test accuracy = 62.95%
Epoch = 39, train accuracy = 61.81%, test accuracy = 62.31%
Epoch = 40, train accuracy = 62.47%, test accuracy = 62.44%
Epoch = 41, train accuracy = 62.91%, test accuracy = 62.44%
Epoch = 42, train accuracy = 61.98%, test accuracy = 61.15%
Epoch = 43, train accuracy = 63.02%, test accuracy = 61.54%
Epoch = 44, train accuracy = 63.85%, test accuracy = 63.33%
Epoch = 45, train accuracy = 63.96%, test accuracy = 62.82%
Epoch = 46, train accuracy = 63.24%, test accuracy = 61.54%
Epoch = 47, train accuracy = 63.02%, test accuracy = 63.21%
Epoch = 48, train accuracy = 65.93%, test accuracy = 64.62%
Epoch = 49, train accuracy = 64.23%, test accuracy = 61.67%
Epoch = 50, train accuracy = 63.79%, test accuracy = 61.79%
Epoch = 51, train accuracy = 63.52%, test accuracy = 62.05%
Epoch = 52, train accuracy = 63.46%, test accuracy = 61.92%
Epoch = 53, train accuracy = 62.91%, test accuracy = 62.69%
Epoch = 54, train accuracy = 64.67%, test accuracy = 63.97%
Epoch = 55, train accuracy = 63.79%, test accuracy = 64.23%
Epoch = 56, train accuracy = 64.29%, test accuracy = 63.97%
Epoch = 57, train accuracy = 64.01%, test accuracy = 61.92%
Epoch = 58, train accuracy = 63.79%, test accuracy = 62.95%
Epoch = 59, train accuracy = 62.47%, test accuracy = 62.82%
Epoch = 60, train accuracy = 63.85%, test accuracy = 61.79%
Epoch = 61, train accuracy = 64.73%, test accuracy = 63.59%
Epoch = 62, train accuracy = 64.01%, test accuracy = 63.46%
Epoch = 63, train accuracy = 64.84%, test accuracy = 63.59%
Epoch = 64, train accuracy = 63.85%, test accuracy = 63.33%
Epoch = 65, train accuracy = 63.02%, test accuracy = 61.54%
Epoch = 66, train accuracy = 64.95%, test accuracy = 64.10%
Epoch = 67, train accuracy = 64.62%, test accuracy = 64.49%
Epoch = 68, train accuracy = 65.05%, test accuracy = 65.38%
Epoch = 69, train accuracy = 63.68%, test accuracy = 63.59%
Epoch = 70, train accuracy = 65.05%, test accuracy = 64.10%
Epoch = 71, train accuracy = 65.00%, test accuracy = 64.74%
Epoch = 72, train accuracy = 64.78%, test accuracy = 63.85%
Epoch = 73, train accuracy = 64.01%, test accuracy = 64.74%
Epoch = 74, train accuracy = 64.78%, test accuracy = 65.26%
Epoch = 75, train accuracy = 64.51%, test accuracy = 64.74%
Epoch = 76, train accuracy = 65.22%, test accuracy = 65.26%
Epoch = 77, train accuracy = 65.55%, test accuracy = 64.87%
Epoch = 78, train accuracy = 63.74%, test accuracy = 63.72%
Epoch = 79, train accuracy = 63.19%, test accuracy = 61.54%
Epoch = 80, train accuracy = 64.84%, test accuracy = 65.13%
Epoch = 81, train accuracy = 65.00%, test accuracy = 65.13%
Epoch = 82, train accuracy = 64.34%, test accuracy = 63.21%
Epoch = 83, train accuracy = 65.00%, test accuracy = 65.77%
Epoch = 84, train accuracy = 66.04%, test accuracy = 65.64%
Epoch = 85, train accuracy = 64.45%, test accuracy = 64.62%
Epoch = 86, train accuracy = 66.81%, test accuracy = 65.90%
Epoch = 87, train accuracy = 67.03%, test accuracy = 66.28%
Epoch = 88, train accuracy = 67.25%, test accuracy = 66.15%
Epoch = 89, train accuracy = 69.01%, test accuracy = 67.82%
Epoch = 90, train accuracy = 65.66%, test accuracy = 66.03%
Epoch = 91, train accuracy = 68.79%, test accuracy = 67.56%
Epoch = 92, train accuracy = 68.52%, test accuracy = 67.05%
Epoch = 93, train accuracy = 68.19%, test accuracy = 68.85%
Epoch = 94, train accuracy = 69.56%, test accuracy = 68.59%
Epoch = 95, train accuracy = 69.56%, test accuracy = 68.85%
Epoch = 96, train accuracy = 70.71%, test accuracy = 68.08%
Epoch = 97, train accuracy = 69.18%, test accuracy = 67.31%
Epoch = 98, train accuracy = 70.05%, test accuracy = 68.33%
Epoch = 99, train accuracy = 69.84%, test accuracy = 69.36%
Epoch = 100, train accuracy = 70.99%, test accuracy = 66.54%
Epoch = 101, train accuracy = 72.14%, test accuracy = 69.49%
Epoch = 102, train accuracy = 72.75%, test accuracy = 69.62%
Epoch = 103, train accuracy = 71.98%, test accuracy = 68.59%
Epoch = 104, train accuracy = 72.36%, test accuracy = 68.59%
Epoch = 105, train accuracy = 71.70%, test accuracy = 67.95%
Epoch = 106, train accuracy = 72.03%, test accuracy = 68.33%
Epoch = 107, train accuracy = 71.21%, test accuracy = 68.21%
Epoch = 108, train accuracy = 70.60%, test accuracy = 68.08%
Epoch = 109, train accuracy = 70.82%, test accuracy = 67.44%
Epoch = 110, train accuracy = 70.55%, test accuracy = 67.18%
Epoch = 111, train accuracy = 69.89%, test accuracy = 66.79%
Epoch = 112, train accuracy = 71.98%, test accuracy = 68.59%
Epoch = 113, train accuracy = 70.55%, test accuracy = 66.28%
Epoch = 114, train accuracy = 66.10%, test accuracy = 62.05%
Epoch = 115, train accuracy = 70.49%, test accuracy = 67.31%
Epoch = 116, train accuracy = 72.97%, test accuracy = 69.49%
Epoch = 117, train accuracy = 74.01%, test accuracy = 68.33%
Epoch = 118, train accuracy = 67.69%, test accuracy = 63.08%
Epoch = 119, train accuracy = 69.78%, test accuracy = 65.90%
Epoch = 120, train accuracy = 71.48%, test accuracy = 67.05%
Epoch = 121, train accuracy = 70.88%, test accuracy = 67.05%
Epoch = 122, train accuracy = 71.43%, test accuracy = 67.56%
Epoch = 123, train accuracy = 69.07%, test accuracy = 64.62%
Epoch = 124, train accuracy = 70.44%, test accuracy = 66.79%
Epoch = 125, train accuracy = 70.33%, test accuracy = 66.79%
Epoch = 126, train accuracy = 66.21%, test accuracy = 61.67%
Epoch = 127, train accuracy = 69.95%, test accuracy = 64.23%
Epoch = 128, train accuracy = 71.76%, test accuracy = 67.05%
Epoch = 129, train accuracy = 69.51%, test accuracy = 65.00%
Epoch = 130, train accuracy = 69.01%, test accuracy = 65.00%
Epoch = 131, train accuracy = 69.29%, test accuracy = 66.03%
Epoch = 132, train accuracy = 69.67%, test accuracy = 65.13%
Epoch = 133, train accuracy = 69.51%, test accuracy = 66.03%
Epoch = 134, train accuracy = 68.68%, test accuracy = 65.26%
Epoch = 135, train accuracy = 71.65%, test accuracy = 67.69%
Epoch = 136, train accuracy = 69.95%, test accuracy = 66.92%
Epoch = 137, train accuracy = 72.14%, test accuracy = 67.44%
Epoch = 138, train accuracy = 70.60%, test accuracy = 67.05%
Epoch = 139, train accuracy = 71.70%, test accuracy = 68.33%
Epoch = 140, train accuracy = 68.02%, test accuracy = 66.28%
Epoch = 141, train accuracy = 69.89%, test accuracy = 66.54%
Epoch = 142, train accuracy = 71.26%, test accuracy = 67.18%
Epoch = 143, train accuracy = 69.73%, test accuracy = 66.28%
Epoch = 144, train accuracy = 70.66%, test accuracy = 66.03%
Epoch = 145, train accuracy = 71.48%, test accuracy = 67.44%
Epoch = 146, train accuracy = 70.60%, test accuracy = 66.15%
Epoch = 147, train accuracy = 71.81%, test accuracy = 67.31%
Epoch = 148, train accuracy = 70.49%, test accuracy = 66.03%
Epoch = 149, train accuracy = 70.44%, test accuracy = 66.28%
Epoch = 150, train accuracy = 71.21%, test accuracy = 66.79%
Epoch = 151, train accuracy = 71.37%, test accuracy = 66.54%
Epoch = 152, train accuracy = 69.95%, test accuracy = 65.26%
Epoch = 153, train accuracy = 72.47%, test accuracy = 67.95%
Epoch = 154, train accuracy = 72.97%, test accuracy = 68.59%
Epoch = 155, train accuracy = 70.93%, test accuracy = 65.90%
Epoch = 156, train accuracy = 73.41%, test accuracy = 68.08%
Epoch = 157, train accuracy = 73.30%, test accuracy = 68.97%
Epoch = 158, train accuracy = 72.03%, test accuracy = 67.95%
Epoch = 159, train accuracy = 74.73%, test accuracy = 68.72%
Epoch = 160, train accuracy = 70.44%, test accuracy = 68.08%
Epoch = 161, train accuracy = 72.97%, test accuracy = 67.69%
Epoch = 162, train accuracy = 73.52%, test accuracy = 68.21%
Epoch = 163, train accuracy = 72.69%, test accuracy = 67.82%
Epoch = 164, train accuracy = 73.24%, test accuracy = 67.44%
Epoch = 165, train accuracy = 69.89%, test accuracy = 64.74%
Epoch = 166, train accuracy = 74.51%, test accuracy = 69.36%
Epoch = 167, train accuracy = 75.16%, test accuracy = 70.38%
Epoch = 168, train accuracy = 68.24%, test accuracy = 63.85%
Epoch = 169, train accuracy = 75.27%, test accuracy = 70.51%
Epoch = 170, train accuracy = 72.80%, test accuracy = 68.21%
Epoch = 171, train accuracy = 70.77%, test accuracy = 67.05%
Epoch = 172, train accuracy = 73.79%, test accuracy = 69.23%
Epoch = 173, train accuracy = 74.78%, test accuracy = 69.36%
Epoch = 174, train accuracy = 71.92%, test accuracy = 67.95%
Epoch = 175, train accuracy = 72.09%, test accuracy = 68.59%
Epoch = 176, train accuracy = 71.32%, test accuracy = 68.21%
Epoch = 177, train accuracy = 73.41%, test accuracy = 68.46%
Epoch = 178, train accuracy = 73.52%, test accuracy = 68.46%
Epoch = 179, train accuracy = 74.89%, test accuracy = 68.21%
Epoch = 180, train accuracy = 71.37%, test accuracy = 67.44%
Epoch = 181, train accuracy = 72.97%, test accuracy = 68.33%
Epoch = 182, train accuracy = 73.13%, test accuracy = 67.82%
Epoch = 183, train accuracy = 70.77%, test accuracy = 66.92%
Epoch = 184, train accuracy = 73.68%, test accuracy = 68.33%
Epoch = 185, train accuracy = 77.03%, test accuracy = 71.15%
Epoch = 186, train accuracy = 73.13%, test accuracy = 67.56%
Epoch = 187, train accuracy = 72.97%, test accuracy = 68.33%
Epoch = 188, train accuracy = 73.85%, test accuracy = 68.46%
Epoch = 189, train accuracy = 76.81%, test accuracy = 70.64%
Epoch = 190, train accuracy = 74.73%, test accuracy = 69.74%
Epoch = 191, train accuracy = 76.26%, test accuracy = 69.62%
Epoch = 192, train accuracy = 75.38%, test accuracy = 69.62%
Epoch = 193, train accuracy = 73.85%, test accuracy = 68.97%
Epoch = 194, train accuracy = 74.23%, test accuracy = 68.72%
Epoch = 195, train accuracy = 75.44%, test accuracy = 70.00%
Epoch = 196, train accuracy = 72.75%, test accuracy = 69.74%
Epoch = 197, train accuracy = 71.70%, test accuracy = 67.44%
Epoch = 198, train accuracy = 74.01%, test accuracy = 69.10%
Epoch = 199, train accuracy = 76.15%, test accuracy = 70.77%
Epoch = 200, train accuracy = 76.37%, test accuracy = 70.51%
Epoch = 201, train accuracy = 74.89%, test accuracy = 68.85%
Epoch = 202, train accuracy = 72.80%, test accuracy = 68.72%
Epoch = 203, train accuracy = 76.81%, test accuracy = 70.90%
Epoch = 204, train accuracy = 74.84%, test accuracy = 69.74%
Epoch = 205, train accuracy = 75.82%, test accuracy = 70.26%
Epoch = 206, train accuracy = 72.53%, test accuracy = 70.38%
Epoch = 207, train accuracy = 75.33%, test accuracy = 69.49%
Epoch = 208, train accuracy = 74.40%, test accuracy = 70.64%
Epoch = 209, train accuracy = 76.43%, test accuracy = 69.23%
Epoch = 210, train accuracy = 73.96%, test accuracy = 67.82%
Epoch = 211, train accuracy = 72.75%, test accuracy = 68.97%
Epoch = 212, train accuracy = 74.40%, test accuracy = 69.10%
Epoch = 213, train accuracy = 73.02%, test accuracy = 68.33%
Epoch = 214, train accuracy = 75.49%, test accuracy = 69.10%
Epoch = 215, train accuracy = 72.09%, test accuracy = 68.72%
Epoch = 216, train accuracy = 75.22%, test accuracy = 70.38%
Epoch = 217, train accuracy = 73.52%, test accuracy = 68.85%
Epoch = 218, train accuracy = 77.20%, test accuracy = 70.13%
Epoch = 219, train accuracy = 76.32%, test accuracy = 70.13%
Epoch = 220, train accuracy = 74.45%, test accuracy = 69.36%
Epoch = 221, train accuracy = 74.07%, test accuracy = 69.49%
Epoch = 222, train accuracy = 75.66%, test accuracy = 70.13%
Epoch = 223, train accuracy = 74.95%, test accuracy = 69.23%
Epoch = 224, train accuracy = 69.78%, test accuracy = 67.05%
Epoch = 225, train accuracy = 73.85%, test accuracy = 67.82%
Epoch = 226, train accuracy = 73.96%, test accuracy = 68.08%
Epoch = 227, train accuracy = 73.85%, test accuracy = 67.56%
Epoch = 228, train accuracy = 72.97%, test accuracy = 68.33%
Epoch = 229, train accuracy = 71.43%, test accuracy = 67.44%
Epoch = 230, train accuracy = 72.75%, test accuracy = 67.82%
Epoch = 231, train accuracy = 76.43%, test accuracy = 69.62%
Epoch = 232, train accuracy = 75.93%, test accuracy = 70.90%
Epoch = 233, train accuracy = 69.95%, test accuracy = 66.41%
Epoch = 234, train accuracy = 72.14%, test accuracy = 68.85%
Epoch = 235, train accuracy = 75.93%, test accuracy = 69.74%
Epoch = 236, train accuracy = 71.70%, test accuracy = 68.08%
Epoch = 237, train accuracy = 74.07%, test accuracy = 69.62%
Epoch = 238, train accuracy = 75.27%, test accuracy = 69.87%
Epoch = 239, train accuracy = 76.10%, test accuracy = 69.87%
Epoch = 240, train accuracy = 76.37%, test accuracy = 69.10%
Epoch = 241, train accuracy = 73.30%, test accuracy = 68.33%
Epoch = 242, train accuracy = 73.30%, test accuracy = 68.97%
Epoch = 243, train accuracy = 74.40%, test accuracy = 68.97%
Epoch = 244, train accuracy = 75.16%, test accuracy = 68.72%
Epoch = 245, train accuracy = 76.15%, test accuracy = 69.10%
Epoch = 246, train accuracy = 75.88%, test accuracy = 70.64%
Epoch = 247, train accuracy = 74.67%, test accuracy = 69.62%
Epoch = 248, train accuracy = 75.93%, test accuracy = 69.36%
Epoch = 249, train accuracy = 72.97%, test accuracy = 67.56%
Epoch = 250, train accuracy = 77.25%, test accuracy = 70.00%
Epoch = 251, train accuracy = 75.11%, test accuracy = 67.44%
Epoch = 252, train accuracy = 75.99%, test accuracy = 70.13%
Epoch = 253, train accuracy = 78.24%, test accuracy = 70.64%
Epoch = 254, train accuracy = 74.67%, test accuracy = 70.26%
Epoch = 255, train accuracy = 74.45%, test accuracy = 68.72%
Epoch = 256, train accuracy = 74.07%, test accuracy = 68.72%
Epoch = 257, train accuracy = 75.00%, test accuracy = 69.10%
Epoch = 258, train accuracy = 76.04%, test accuracy = 70.64%
Epoch = 259, train accuracy = 75.93%, test accuracy = 70.00%
Epoch = 260, train accuracy = 73.52%, test accuracy = 68.59%
Epoch = 261, train accuracy = 75.99%, test accuracy = 70.90%
Epoch = 262, train accuracy = 75.11%, test accuracy = 69.23%
Epoch = 263, train accuracy = 74.29%, test accuracy = 69.49%
Epoch = 264, train accuracy = 75.77%, test accuracy = 69.87%
Epoch = 265, train accuracy = 74.84%, test accuracy = 69.87%
Epoch = 266, train accuracy = 75.82%, test accuracy = 70.51%
Epoch = 267, train accuracy = 75.16%, test accuracy = 71.03%
Epoch = 268, train accuracy = 73.96%, test accuracy = 68.33%
Epoch = 269, train accuracy = 76.04%, test accuracy = 70.13%
Epoch = 270, train accuracy = 76.15%, test accuracy = 71.03%
Epoch = 271, train accuracy = 75.16%, test accuracy = 71.03%
Epoch = 272, train accuracy = 76.10%, test accuracy = 69.87%
Epoch = 273, train accuracy = 74.84%, test accuracy = 69.49%
Epoch = 274, train accuracy = 76.26%, test accuracy = 70.26%
Epoch = 275, train accuracy = 75.66%, test accuracy = 69.49%
Epoch = 276, train accuracy = 75.11%, test accuracy = 70.13%
Epoch = 277, train accuracy = 75.38%, test accuracy = 67.56%
Epoch = 278, train accuracy = 75.66%, test accuracy = 69.87%
Epoch = 279, train accuracy = 74.62%, test accuracy = 69.23%
Epoch = 280, train accuracy = 77.86%, test accuracy = 70.64%
Epoch = 281, train accuracy = 72.36%, test accuracy = 66.41%
Epoch = 282, train accuracy = 76.70%, test accuracy = 70.13%
Epoch = 283, train accuracy = 76.32%, test accuracy = 69.74%
Epoch = 284, train accuracy = 77.09%, test accuracy = 68.97%
Epoch = 285, train accuracy = 76.92%, test accuracy = 69.49%
Epoch = 286, train accuracy = 72.91%, test accuracy = 68.59%
Epoch = 287, train accuracy = 75.44%, test accuracy = 68.08%
Epoch = 288, train accuracy = 75.49%, test accuracy = 70.77%
Epoch = 289, train accuracy = 76.48%, test accuracy = 71.54%
Epoch = 290, train accuracy = 75.82%, test accuracy = 70.26%
Epoch = 291, train accuracy = 76.54%, test accuracy = 70.00%
Epoch = 292, train accuracy = 75.27%, test accuracy = 70.38%
Epoch = 293, train accuracy = 76.04%, test accuracy = 69.23%
Epoch = 294, train accuracy = 74.40%, test accuracy = 69.49%
Epoch = 295, train accuracy = 72.64%, test accuracy = 67.82%
Epoch = 296, train accuracy = 75.33%, test accuracy = 69.49%
Epoch = 297, train accuracy = 75.82%, test accuracy = 68.85%
Epoch = 298, train accuracy = 75.16%, test accuracy = 69.87%
Epoch = 299, train accuracy = 74.51%, test accuracy = 70.38%
Epoch = 300, train accuracy = 74.62%, test accuracy = 68.85%
Epoch = 301, train accuracy = 73.08%, test accuracy = 67.31%
Epoch = 302, train accuracy = 77.58%, test accuracy = 70.51%
Epoch = 303, train accuracy = 77.25%, test accuracy = 69.36%
Epoch = 304, train accuracy = 74.51%, test accuracy = 68.21%
Epoch = 305, train accuracy = 75.71%, test accuracy = 69.62%
Epoch = 306, train accuracy = 74.29%, test accuracy = 68.85%
Epoch = 307, train accuracy = 74.12%, test accuracy = 68.46%
Epoch = 308, train accuracy = 75.66%, test accuracy = 67.18%
Epoch = 309, train accuracy = 74.07%, test accuracy = 68.08%
Epoch = 310, train accuracy = 74.67%, test accuracy = 69.10%
Epoch = 311, train accuracy = 75.77%, test accuracy = 70.13%
Epoch = 312, train accuracy = 76.43%, test accuracy = 68.33%
Epoch = 313, train accuracy = 76.98%, test accuracy = 71.41%
Epoch = 314, train accuracy = 75.49%, test accuracy = 68.08%
Epoch = 315, train accuracy = 75.71%, test accuracy = 69.23%
Epoch = 316, train accuracy = 74.23%, test accuracy = 68.72%
Epoch = 317, train accuracy = 75.05%, test accuracy = 67.95%
Epoch = 318, train accuracy = 75.88%, test accuracy = 68.21%
Epoch = 319, train accuracy = 75.00%, test accuracy = 67.18%
Epoch = 320, train accuracy = 77.09%, test accuracy = 68.59%
Epoch = 321, train accuracy = 76.37%, test accuracy = 69.36%
Epoch = 322, train accuracy = 73.63%, test accuracy = 68.33%
Epoch = 323, train accuracy = 72.86%, test accuracy = 66.67%
Epoch = 324, train accuracy = 71.92%, test accuracy = 67.95%
Epoch = 325, train accuracy = 72.47%, test accuracy = 66.28%
Epoch = 326, train accuracy = 77.47%, test accuracy = 71.15%
Epoch = 327, train accuracy = 76.26%, test accuracy = 68.72%
Epoch = 328, train accuracy = 76.10%, test accuracy = 68.59%
Epoch = 329, train accuracy = 76.10%, test accuracy = 68.72%
Epoch = 330, train accuracy = 73.74%, test accuracy = 67.56%
Epoch = 331, train accuracy = 77.20%, test accuracy = 67.95%
Epoch = 332, train accuracy = 75.88%, test accuracy = 68.72%
Epoch = 333, train accuracy = 76.92%, test accuracy = 71.41%
Epoch = 334, train accuracy = 74.78%, test accuracy = 67.05%
Epoch = 335, train accuracy = 75.71%, test accuracy = 70.51%
Epoch = 336, train accuracy = 73.30%, test accuracy = 67.82%
Epoch = 337, train accuracy = 76.70%, test accuracy = 68.72%
Epoch = 338, train accuracy = 72.20%, test accuracy = 66.03%
Epoch = 339, train accuracy = 75.99%, test accuracy = 68.97%
Epoch = 340, train accuracy = 77.14%, test accuracy = 70.77%
Epoch = 341, train accuracy = 74.29%, test accuracy = 68.72%
Epoch = 342, train accuracy = 76.76%, test accuracy = 68.59%
Epoch = 343, train accuracy = 75.44%, test accuracy = 67.69%
Epoch = 344, train accuracy = 76.59%, test accuracy = 68.33%
Epoch = 345, train accuracy = 75.82%, test accuracy = 68.59%
Epoch = 346, train accuracy = 75.49%, test accuracy = 68.08%
Epoch = 347, train accuracy = 77.31%, test accuracy = 70.38%
Epoch = 348, train accuracy = 75.11%, test accuracy = 69.87%
Epoch = 349, train accuracy = 75.00%, test accuracy = 67.95%
Epoch = 350, train accuracy = 77.91%, test accuracy = 69.87%
Epoch = 351, train accuracy = 74.89%, test accuracy = 67.31%
Epoch = 352, train accuracy = 76.43%, test accuracy = 68.33%
Epoch = 353, train accuracy = 75.55%, test accuracy = 69.87%
Epoch = 354, train accuracy = 75.00%, test accuracy = 67.95%
Epoch = 355, train accuracy = 75.77%, test accuracy = 68.85%
Epoch = 356, train accuracy = 76.26%, test accuracy = 68.33%
Epoch = 357, train accuracy = 78.46%, test accuracy = 70.00%
Epoch = 358, train accuracy = 75.71%, test accuracy = 69.62%
Epoch = 359, train accuracy = 76.43%, test accuracy = 69.62%
Epoch = 360, train accuracy = 76.54%, test accuracy = 69.87%
Epoch = 361, train accuracy = 71.10%, test accuracy = 65.90%
Epoch = 362, train accuracy = 75.99%, test accuracy = 68.97%
Epoch = 363, train accuracy = 76.04%, test accuracy = 67.69%
Epoch = 364, train accuracy = 77.80%, test accuracy = 68.33%
Epoch = 365, train accuracy = 71.81%, test accuracy = 66.67%
Epoch = 366, train accuracy = 74.40%, test accuracy = 68.97%
Epoch = 367, train accuracy = 78.74%, test accuracy = 69.62%
Epoch = 368, train accuracy = 76.32%, test accuracy = 69.36%
Epoch = 369, train accuracy = 76.98%, test accuracy = 68.72%
Epoch = 370, train accuracy = 76.15%, test accuracy = 68.59%
Epoch = 371, train accuracy = 77.53%, test accuracy = 68.59%
Epoch = 372, train accuracy = 79.67%, test accuracy = 69.62%
Epoch = 373, train accuracy = 79.29%, test accuracy = 69.49%
Epoch = 374, train accuracy = 77.53%, test accuracy = 70.13%
Epoch = 375, train accuracy = 77.53%, test accuracy = 69.62%
Epoch = 376, train accuracy = 78.02%, test accuracy = 68.72%
Epoch = 377, train accuracy = 77.97%, test accuracy = 69.62%
Epoch = 378, train accuracy = 77.80%, test accuracy = 68.85%
Epoch = 379, train accuracy = 77.86%, test accuracy = 69.36%
Epoch = 380, train accuracy = 76.92%, test accuracy = 69.49%
Epoch = 381, train accuracy = 77.75%, test accuracy = 69.62%
Epoch = 382, train accuracy = 76.10%, test accuracy = 67.95%
Epoch = 383, train accuracy = 76.48%, test accuracy = 67.56%
Epoch = 384, train accuracy = 79.78%, test accuracy = 70.13%
Epoch = 385, train accuracy = 77.80%, test accuracy = 67.56%
Epoch = 386, train accuracy = 77.14%, test accuracy = 67.69%
Epoch = 387, train accuracy = 77.25%, test accuracy = 68.46%
Epoch = 388, train accuracy = 78.24%, test accuracy = 68.97%
Epoch = 389, train accuracy = 79.78%, test accuracy = 68.21%
Epoch = 390, train accuracy = 78.90%, test accuracy = 69.62%
Epoch = 391, train accuracy = 77.47%, test accuracy = 67.56%
Epoch = 392, train accuracy = 73.13%, test accuracy = 67.95%
Epoch = 393, train accuracy = 81.43%, test accuracy = 69.74%
Epoch = 394, train accuracy = 65.27%, test accuracy = 58.59%
Epoch = 395, train accuracy = 77.42%, test accuracy = 68.72%
Epoch = 396, train accuracy = 79.23%, test accuracy = 67.82%
Epoch = 397, train accuracy = 79.78%, test accuracy = 70.38%
Epoch = 398, train accuracy = 76.87%, test accuracy = 67.44%
Epoch = 399, train accuracy = 77.42%, test accuracy = 67.82%
Epoch = 400, train accuracy = 76.21%, test accuracy = 68.21%
Epoch = 401, train accuracy = 75.38%, test accuracy = 68.08%
Epoch = 402, train accuracy = 77.31%, test accuracy = 68.46%
Epoch = 403, train accuracy = 76.37%, test accuracy = 66.41%
Epoch = 404, train accuracy = 78.41%, test accuracy = 69.62%
Epoch = 405, train accuracy = 79.12%, test accuracy = 67.95%
Epoch = 406, train accuracy = 79.84%, test accuracy = 69.62%
Epoch = 407, train accuracy = 77.25%, test accuracy = 69.49%
Epoch = 408, train accuracy = 77.14%, test accuracy = 67.82%
Epoch = 409, train accuracy = 78.08%, test accuracy = 68.08%
Epoch = 410, train accuracy = 78.13%, test accuracy = 67.95%
Epoch = 411, train accuracy = 80.77%, test accuracy = 68.97%
Epoch = 412, train accuracy = 79.40%, test accuracy = 69.62%
Epoch = 413, train accuracy = 77.80%, test accuracy = 68.46%
Epoch = 414, train accuracy = 76.26%, test accuracy = 68.85%
Epoch = 415, train accuracy = 77.75%, test accuracy = 66.15%
Epoch = 416, train accuracy = 80.05%, test accuracy = 70.00%
Epoch = 417, train accuracy = 75.27%, test accuracy = 67.44%
Epoch = 418, train accuracy = 79.40%, test accuracy = 69.10%
Epoch = 419, train accuracy = 78.35%, test accuracy = 68.72%
Epoch = 420, train accuracy = 75.60%, test accuracy = 67.69%
Epoch = 421, train accuracy = 78.30%, test accuracy = 68.21%
Epoch = 422, train accuracy = 78.52%, test accuracy = 67.44%
Epoch = 423, train accuracy = 77.25%, test accuracy = 68.59%
Epoch = 424, train accuracy = 79.34%, test accuracy = 68.85%
Epoch = 425, train accuracy = 74.78%, test accuracy = 65.90%
Epoch = 426, train accuracy = 76.87%, test accuracy = 67.05%
Epoch = 427, train accuracy = 78.68%, test accuracy = 68.33%
Epoch = 428, train accuracy = 79.40%, test accuracy = 68.59%
Epoch = 429, train accuracy = 81.10%, test accuracy = 69.62%
Epoch = 430, train accuracy = 64.12%, test accuracy = 57.82%
Epoch = 431, train accuracy = 78.24%, test accuracy = 68.33%
Epoch = 432, train accuracy = 80.22%, test accuracy = 68.97%
Epoch = 433, train accuracy = 76.04%, test accuracy = 70.38%
Epoch = 434, train accuracy = 78.74%, test accuracy = 68.59%
Epoch = 435, train accuracy = 81.21%, test accuracy = 68.72%
Epoch = 436, train accuracy = 80.05%, test accuracy = 67.82%
Epoch = 437, train accuracy = 81.48%, test accuracy = 69.87%
Epoch = 438, train accuracy = 80.71%, test accuracy = 69.62%
Epoch = 439, train accuracy = 80.44%, test accuracy = 69.62%
Epoch = 440, train accuracy = 80.22%, test accuracy = 68.85%
Epoch = 441, train accuracy = 78.79%, test accuracy = 66.79%
Epoch = 442, train accuracy = 79.51%, test accuracy = 68.46%
Epoch = 443, train accuracy = 77.25%, test accuracy = 67.31%
Epoch = 444, train accuracy = 78.30%, test accuracy = 68.08%
Epoch = 445, train accuracy = 78.46%, test accuracy = 68.46%
Epoch = 446, train accuracy = 79.29%, test accuracy = 69.23%
Epoch = 447, train accuracy = 78.13%, test accuracy = 68.08%
Epoch = 448, train accuracy = 78.52%, test accuracy = 69.10%
Epoch = 449, train accuracy = 79.23%, test accuracy = 68.21%
Epoch = 450, train accuracy = 78.96%, test accuracy = 68.59%
Epoch = 451, train accuracy = 80.22%, test accuracy = 67.44%
Epoch = 452, train accuracy = 77.36%, test accuracy = 68.59%
Epoch = 453, train accuracy = 81.54%, test accuracy = 69.36%
Epoch = 454, train accuracy = 80.88%, test accuracy = 69.74%
Epoch = 455, train accuracy = 77.86%, test accuracy = 68.08%
Epoch = 456, train accuracy = 79.23%, test accuracy = 68.21%
Epoch = 457, train accuracy = 75.44%, test accuracy = 66.03%
Epoch = 458, train accuracy = 76.65%, test accuracy = 67.31%
Epoch = 459, train accuracy = 79.51%, test accuracy = 68.72%
Epoch = 460, train accuracy = 77.53%, test accuracy = 66.79%
Epoch = 461, train accuracy = 80.55%, test accuracy = 69.36%
Epoch = 462, train accuracy = 79.12%, test accuracy = 67.44%
Epoch = 463, train accuracy = 80.77%, test accuracy = 70.64%
Epoch = 464, train accuracy = 77.58%, test accuracy = 68.08%
Epoch = 465, train accuracy = 79.40%, test accuracy = 68.21%
Epoch = 466, train accuracy = 80.22%, test accuracy = 68.72%
Epoch = 467, train accuracy = 76.76%, test accuracy = 68.46%
Epoch = 468, train accuracy = 80.88%, test accuracy = 67.56%
Epoch = 469, train accuracy = 75.27%, test accuracy = 67.69%
Epoch = 470, train accuracy = 78.74%, test accuracy = 67.56%
Epoch = 471, train accuracy = 79.34%, test accuracy = 69.36%
Epoch = 472, train accuracy = 76.87%, test accuracy = 68.08%
Epoch = 473, train accuracy = 81.32%, test accuracy = 70.13%
Epoch = 474, train accuracy = 80.66%, test accuracy = 69.49%
Epoch = 475, train accuracy = 79.84%, test accuracy = 70.00%
Epoch = 476, train accuracy = 78.41%, test accuracy = 67.95%
Epoch = 477, train accuracy = 80.66%, test accuracy = 68.85%
Epoch = 478, train accuracy = 80.44%, test accuracy = 67.82%
Epoch = 479, train accuracy = 80.93%, test accuracy = 69.23%
Epoch = 480, train accuracy = 80.38%, test accuracy = 68.59%
Epoch = 481, train accuracy = 79.67%, test accuracy = 67.56%
Epoch = 482, train accuracy = 80.49%, test accuracy = 67.44%
Epoch = 483, train accuracy = 77.64%, test accuracy = 66.79%
Epoch = 484, train accuracy = 78.63%, test accuracy = 66.54%
Epoch = 485, train accuracy = 80.27%, test accuracy = 67.69%
Epoch = 486, train accuracy = 80.44%, test accuracy = 70.13%
Epoch = 487, train accuracy = 79.62%, test accuracy = 67.18%
Epoch = 488, train accuracy = 81.32%, test accuracy = 69.49%
Epoch = 489, train accuracy = 83.13%, test accuracy = 69.87%
Epoch = 490, train accuracy = 76.04%, test accuracy = 68.21%
Epoch = 491, train accuracy = 78.30%, test accuracy = 67.82%
Epoch = 492, train accuracy = 80.55%, test accuracy = 67.95%
Epoch = 493, train accuracy = 82.25%, test accuracy = 69.36%
Epoch = 494, train accuracy = 81.59%, test accuracy = 68.72%
Epoch = 495, train accuracy = 82.75%, test accuracy = 68.46%
Epoch = 496, train accuracy = 84.07%, test accuracy = 69.36%
Epoch = 497, train accuracy = 80.88%, test accuracy = 67.95%
Epoch = 498, train accuracy = 81.21%, test accuracy = 68.46%
Epoch = 499, train accuracy = 81.26%, test accuracy = 67.82%
Epoch = 500, train accuracy = 80.60%, test accuracy = 69.49%
Epoch = 501, train accuracy = 79.78%, test accuracy = 68.33%
Epoch = 502, train accuracy = 77.80%, test accuracy = 67.31%
Epoch = 503, train accuracy = 83.46%, test accuracy = 69.74%
Epoch = 504, train accuracy = 76.70%, test accuracy = 66.03%
Epoch = 505, train accuracy = 82.20%, test accuracy = 69.10%
Epoch = 506, train accuracy = 80.77%, test accuracy = 68.33%
Epoch = 507, train accuracy = 82.25%, test accuracy = 70.38%
Epoch = 508, train accuracy = 83.63%, test accuracy = 69.87%
Epoch = 509, train accuracy = 83.08%, test accuracy = 69.87%
Epoch = 510, train accuracy = 83.57%, test accuracy = 69.74%
Epoch = 511, train accuracy = 79.18%, test accuracy = 67.56%
Epoch = 512, train accuracy = 81.48%, test accuracy = 68.21%
Epoch = 513, train accuracy = 76.54%, test accuracy = 65.51%
Epoch = 514, train accuracy = 80.11%, test accuracy = 67.56%
Epoch = 515, train accuracy = 79.73%, test accuracy = 66.41%
Epoch = 516, train accuracy = 78.30%, test accuracy = 68.21%
Epoch = 517, train accuracy = 77.47%, test accuracy = 66.15%
Epoch = 518, train accuracy = 82.31%, test accuracy = 68.85%
Epoch = 519, train accuracy = 80.77%, test accuracy = 67.05%
Epoch = 520, train accuracy = 79.84%, test accuracy = 68.97%
Epoch = 521, train accuracy = 82.20%, test accuracy = 69.10%
Epoch = 522, train accuracy = 80.05%, test accuracy = 67.95%
Epoch = 523, train accuracy = 83.41%, test accuracy = 69.74%
Epoch = 524, train accuracy = 83.57%, test accuracy = 70.13%
Epoch = 525, train accuracy = 81.76%, test accuracy = 70.00%
Epoch = 526, train accuracy = 81.21%, test accuracy = 68.21%
Epoch = 527, train accuracy = 83.13%, test accuracy = 69.49%
Epoch = 528, train accuracy = 76.32%, test accuracy = 66.79%
Epoch = 529, train accuracy = 77.53%, test accuracy = 66.15%
Epoch = 530, train accuracy = 80.88%, test accuracy = 68.21%
Epoch = 531, train accuracy = 81.10%, test accuracy = 69.10%
Epoch = 532, train accuracy = 81.21%, test accuracy = 69.62%
Epoch = 533, train accuracy = 77.20%, test accuracy = 65.90%
Epoch = 534, train accuracy = 82.69%, test accuracy = 68.08%
Epoch = 535, train accuracy = 78.46%, test accuracy = 67.69%
Epoch = 536, train accuracy = 82.47%, test accuracy = 69.87%
Epoch = 537, train accuracy = 82.31%, test accuracy = 69.49%
Epoch = 538, train accuracy = 82.03%, test accuracy = 68.21%
Epoch = 539, train accuracy = 80.11%, test accuracy = 68.21%
Epoch = 540, train accuracy = 82.25%, test accuracy = 68.33%
Epoch = 541, train accuracy = 82.64%, test accuracy = 68.72%
Epoch = 542, train accuracy = 80.33%, test accuracy = 66.54%
Epoch = 543, train accuracy = 80.55%, test accuracy = 68.59%
Epoch = 544, train accuracy = 82.42%, test accuracy = 69.62%
Epoch = 545, train accuracy = 81.81%, test accuracy = 67.31%
Epoch = 546, train accuracy = 74.51%, test accuracy = 63.85%
Epoch = 547, train accuracy = 84.23%, test accuracy = 69.10%
Epoch = 548, train accuracy = 81.26%, test accuracy = 68.08%
Epoch = 549, train accuracy = 82.20%, test accuracy = 68.72%
Epoch = 550, train accuracy = 84.12%, test accuracy = 68.46%
Epoch = 551, train accuracy = 83.02%, test accuracy = 69.62%
Epoch = 552, train accuracy = 80.27%, test accuracy = 67.69%
Epoch = 553, train accuracy = 79.73%, test accuracy = 67.44%
Epoch = 554, train accuracy = 80.38%, test accuracy = 68.08%
Epoch = 555, train accuracy = 82.03%, test accuracy = 67.18%
Epoch = 556, train accuracy = 80.27%, test accuracy = 69.10%
Epoch = 557, train accuracy = 78.08%, test accuracy = 65.64%
Epoch = 558, train accuracy = 78.08%, test accuracy = 66.92%
Epoch = 559, train accuracy = 85.88%, test accuracy = 69.62%
Epoch = 560, train accuracy = 81.04%, test accuracy = 68.72%
Epoch = 561, train accuracy = 83.46%, test accuracy = 69.87%
Epoch = 562, train accuracy = 83.52%, test accuracy = 69.49%
Epoch = 563, train accuracy = 82.64%, test accuracy = 66.92%
Epoch = 564, train accuracy = 83.13%, test accuracy = 68.46%
Epoch = 565, train accuracy = 79.95%, test accuracy = 67.95%
Epoch = 566, train accuracy = 85.00%, test accuracy = 69.10%
Epoch = 567, train accuracy = 85.27%, test accuracy = 68.33%
Epoch = 568, train accuracy = 81.54%, test accuracy = 68.59%
Epoch = 569, train accuracy = 82.64%, test accuracy = 69.49%
Epoch = 570, train accuracy = 85.05%, test accuracy = 70.13%
Epoch = 571, train accuracy = 83.52%, test accuracy = 70.26%
Epoch = 572, train accuracy = 83.02%, test accuracy = 68.85%
Epoch = 573, train accuracy = 81.92%, test accuracy = 68.33%
Epoch = 574, train accuracy = 84.62%, test accuracy = 69.62%
Epoch = 575, train accuracy = 81.54%, test accuracy = 68.08%
Epoch = 576, train accuracy = 83.46%, test accuracy = 69.10%
Epoch = 577, train accuracy = 86.04%, test accuracy = 69.10%
Epoch = 578, train accuracy = 80.82%, test accuracy = 67.31%
Epoch = 579, train accuracy = 82.86%, test accuracy = 69.10%
Epoch = 580, train accuracy = 82.31%, test accuracy = 67.95%
Epoch = 581, train accuracy = 80.00%, test accuracy = 66.67%
Epoch = 582, train accuracy = 80.93%, test accuracy = 68.59%
Epoch = 583, train accuracy = 81.48%, test accuracy = 70.26%
Epoch = 584, train accuracy = 79.89%, test accuracy = 67.18%
Epoch = 585, train accuracy = 81.54%, test accuracy = 69.62%
Epoch = 586, train accuracy = 81.59%, test accuracy = 69.10%
Epoch = 587, train accuracy = 80.71%, test accuracy = 66.79%
Epoch = 588, train accuracy = 82.47%, test accuracy = 68.85%
Epoch = 589, train accuracy = 73.74%, test accuracy = 66.79%
Epoch = 590, train accuracy = 83.96%, test accuracy = 69.49%
Epoch = 591, train accuracy = 82.69%, test accuracy = 68.59%
Epoch = 592, train accuracy = 84.12%, test accuracy = 68.59%
Epoch = 593, train accuracy = 82.36%, test accuracy = 68.97%
Epoch = 594, train accuracy = 81.65%, test accuracy = 67.69%
Epoch = 595, train accuracy = 84.95%, test accuracy = 70.38%
Epoch = 596, train accuracy = 84.84%, test accuracy = 69.74%
Epoch = 597, train accuracy = 83.46%, test accuracy = 69.36%
Epoch = 598, train accuracy = 84.45%, test accuracy = 70.51%
Epoch = 599, train accuracy = 85.71%, test accuracy = 69.49%
Epoch = 600, train accuracy = 83.46%, test accuracy = 70.00%
Epoch = 601, train accuracy = 82.64%, test accuracy = 68.21%
Epoch = 602, train accuracy = 87.36%, test accuracy = 70.77%
Epoch = 603, train accuracy = 84.29%, test accuracy = 69.74%
Epoch = 604, train accuracy = 85.82%, test accuracy = 68.72%
Epoch = 605, train accuracy = 84.95%, test accuracy = 67.82%
Epoch = 606, train accuracy = 81.32%, test accuracy = 67.95%
Epoch = 607, train accuracy = 82.69%, test accuracy = 67.82%
Epoch = 608, train accuracy = 78.74%, test accuracy = 68.59%
Epoch = 609, train accuracy = 77.31%, test accuracy = 66.79%
Epoch = 610, train accuracy = 82.03%, test accuracy = 68.08%
Epoch = 611, train accuracy = 83.30%, test accuracy = 68.46%
Epoch = 612, train accuracy = 83.79%, test accuracy = 68.85%
Epoch = 613, train accuracy = 85.66%, test accuracy = 69.23%
Epoch = 614, train accuracy = 84.95%, test accuracy = 70.38%
Epoch = 615, train accuracy = 83.90%, test accuracy = 69.36%
Epoch = 616, train accuracy = 84.84%, test accuracy = 69.49%
Epoch = 617, train accuracy = 83.24%, test accuracy = 68.46%
Epoch = 618, train accuracy = 84.34%, test accuracy = 68.46%
Epoch = 619, train accuracy = 82.03%, test accuracy = 66.15%
Epoch = 620, train accuracy = 82.80%, test accuracy = 69.10%
Epoch = 621, train accuracy = 81.81%, test accuracy = 68.21%
Epoch = 622, train accuracy = 82.53%, test accuracy = 68.33%
Epoch = 623, train accuracy = 81.54%, test accuracy = 67.18%
Epoch = 624, train accuracy = 83.90%, test accuracy = 68.46%
Epoch = 625, train accuracy = 82.36%, test accuracy = 67.69%
Epoch = 626, train accuracy = 84.67%, test accuracy = 68.46%
Epoch = 627, train accuracy = 83.57%, test accuracy = 66.92%
Epoch = 628, train accuracy = 83.96%, test accuracy = 67.82%
Epoch = 629, train accuracy = 80.55%, test accuracy = 67.31%
Epoch = 630, train accuracy = 85.16%, test accuracy = 69.10%
Epoch = 631, train accuracy = 84.51%, test accuracy = 68.59%
Epoch = 632, train accuracy = 83.35%, test accuracy = 68.08%
Epoch = 633, train accuracy = 85.38%, test accuracy = 68.97%
Epoch = 634, train accuracy = 82.14%, test accuracy = 66.92%
Epoch = 635, train accuracy = 79.07%, test accuracy = 67.05%
Epoch = 636, train accuracy = 80.77%, test accuracy = 65.26%
Epoch = 637, train accuracy = 82.20%, test accuracy = 67.69%
Epoch = 638, train accuracy = 85.99%, test accuracy = 69.23%
Epoch = 639, train accuracy = 82.80%, test accuracy = 67.95%
Epoch = 640, train accuracy = 82.69%, test accuracy = 67.31%
Epoch = 641, train accuracy = 81.65%, test accuracy = 67.44%
Epoch = 642, train accuracy = 86.10%, test accuracy = 69.74%
Epoch = 643, train accuracy = 87.42%, test accuracy = 68.59%
Epoch = 644, train accuracy = 85.49%, test accuracy = 69.49%
Epoch = 645, train accuracy = 80.11%, test accuracy = 68.97%
Epoch = 646, train accuracy = 85.05%, test accuracy = 67.18%
Epoch = 647, train accuracy = 82.14%, test accuracy = 67.44%
Epoch = 648, train accuracy = 83.41%, test accuracy = 67.31%
Epoch = 649, train accuracy = 82.86%, test accuracy = 66.67%
Epoch = 650, train accuracy = 83.68%, test accuracy = 68.72%
Epoch = 651, train accuracy = 84.62%, test accuracy = 68.33%
Epoch = 652, train accuracy = 85.71%, test accuracy = 69.10%
Epoch = 653, train accuracy = 84.73%, test accuracy = 66.92%
Epoch = 654, train accuracy = 87.14%, test accuracy = 69.74%
Epoch = 655, train accuracy = 86.70%, test accuracy = 67.82%
Epoch = 656, train accuracy = 87.31%, test accuracy = 69.36%
Epoch = 657, train accuracy = 88.02%, test accuracy = 70.38%
Epoch = 658, train accuracy = 87.42%, test accuracy = 70.26%
Epoch = 659, train accuracy = 83.68%, test accuracy = 67.31%
Epoch = 660, train accuracy = 87.86%, test accuracy = 69.36%
Epoch = 661, train accuracy = 81.32%, test accuracy = 66.15%
Epoch = 662, train accuracy = 86.32%, test accuracy = 68.59%
Epoch = 663, train accuracy = 88.02%, test accuracy = 68.85%
Epoch = 664, train accuracy = 86.04%, test accuracy = 68.72%
Epoch = 665, train accuracy = 85.77%, test accuracy = 67.95%
Epoch = 666, train accuracy = 86.37%, test accuracy = 67.44%
Epoch = 667, train accuracy = 86.76%, test accuracy = 70.13%
Epoch = 668, train accuracy = 87.75%, test accuracy = 68.46%
Epoch = 669, train accuracy = 83.46%, test accuracy = 69.49%
Epoch = 670, train accuracy = 82.75%, test accuracy = 68.72%
Epoch = 671, train accuracy = 85.99%, test accuracy = 68.85%
Epoch = 672, train accuracy = 88.24%, test accuracy = 68.85%
Epoch = 673, train accuracy = 84.45%, test accuracy = 67.05%
Epoch = 674, train accuracy = 83.08%, test accuracy = 67.44%
Epoch = 675, train accuracy = 86.54%, test accuracy = 67.69%
Epoch = 676, train accuracy = 87.69%, test accuracy = 67.56%
Epoch = 677, train accuracy = 85.93%, test accuracy = 69.10%
Epoch = 678, train accuracy = 88.13%, test accuracy = 67.95%
Epoch = 679, train accuracy = 87.75%, test accuracy = 68.85%
Epoch = 680, train accuracy = 87.97%, test accuracy = 68.59%
Epoch = 681, train accuracy = 75.33%, test accuracy = 63.72%
Epoch = 682, train accuracy = 84.62%, test accuracy = 67.82%
Epoch = 683, train accuracy = 86.87%, test accuracy = 67.18%
Epoch = 684, train accuracy = 86.26%, test accuracy = 67.95%
Epoch = 685, train accuracy = 81.92%, test accuracy = 67.31%
Epoch = 686, train accuracy = 77.09%, test accuracy = 63.33%
Epoch = 687, train accuracy = 83.63%, test accuracy = 65.13%
Epoch = 688, train accuracy = 83.08%, test accuracy = 67.18%
Epoch = 689, train accuracy = 86.48%, test accuracy = 68.72%
Epoch = 690, train accuracy = 86.65%, test accuracy = 69.49%
Epoch = 691, train accuracy = 88.52%, test accuracy = 67.95%
Epoch = 692, train accuracy = 89.40%, test accuracy = 69.49%
Epoch = 693, train accuracy = 87.20%, test accuracy = 68.21%
Epoch = 694, train accuracy = 86.59%, test accuracy = 68.85%
Epoch = 695, train accuracy = 88.08%, test accuracy = 67.95%
Epoch = 696, train accuracy = 85.11%, test accuracy = 67.44%
Epoch = 697, train accuracy = 85.22%, test accuracy = 68.46%
Epoch = 698, train accuracy = 86.87%, test accuracy = 68.72%
Epoch = 699, train accuracy = 86.15%, test accuracy = 67.44%
Epoch = 700, train accuracy = 88.63%, test accuracy = 67.95%
Epoch = 701, train accuracy = 86.54%, test accuracy = 69.23%
Epoch = 702, train accuracy = 90.05%, test accuracy = 68.21%
Epoch = 703, train accuracy = 73.63%, test accuracy = 64.23%
Epoch = 704, train accuracy = 87.69%, test accuracy = 68.33%
Epoch = 705, train accuracy = 87.42%, test accuracy = 68.46%
Epoch = 706, train accuracy = 86.48%, test accuracy = 69.36%
Epoch = 707, train accuracy = 81.10%, test accuracy = 66.54%
Epoch = 708, train accuracy = 79.56%, test accuracy = 64.74%
Epoch = 709, train accuracy = 86.92%, test accuracy = 68.85%
Epoch = 710, train accuracy = 87.69%, test accuracy = 68.72%
Epoch = 711, train accuracy = 87.14%, test accuracy = 68.33%
Epoch = 712, train accuracy = 83.79%, test accuracy = 67.31%
Epoch = 713, train accuracy = 80.93%, test accuracy = 66.79%
Epoch = 714, train accuracy = 88.41%, test accuracy = 68.85%
Epoch = 715, train accuracy = 86.15%, test accuracy = 67.56%
Epoch = 716, train accuracy = 86.98%, test accuracy = 68.46%
Epoch = 717, train accuracy = 89.40%, test accuracy = 68.59%
Epoch = 718, train accuracy = 88.85%, test accuracy = 68.33%
Epoch = 719, train accuracy = 89.23%, test accuracy = 68.85%
Epoch = 720, train accuracy = 80.93%, test accuracy = 67.44%
Epoch = 721, train accuracy = 84.95%, test accuracy = 67.31%
Epoch = 722, train accuracy = 88.41%, test accuracy = 69.36%
Epoch = 723, train accuracy = 88.52%, test accuracy = 67.44%
Epoch = 724, train accuracy = 87.58%, test accuracy = 68.97%
Epoch = 725, train accuracy = 88.30%, test accuracy = 68.72%
Epoch = 726, train accuracy = 87.97%, test accuracy = 67.69%
Epoch = 727, train accuracy = 89.73%, test accuracy = 68.08%
Epoch = 728, train accuracy = 89.56%, test accuracy = 69.49%
Epoch = 729, train accuracy = 87.75%, test accuracy = 68.08%
Epoch = 730, train accuracy = 88.35%, test accuracy = 69.62%
Epoch = 731, train accuracy = 86.48%, test accuracy = 67.31%
Epoch = 732, train accuracy = 85.44%, test accuracy = 67.95%
Epoch = 733, train accuracy = 88.13%, test accuracy = 69.74%
Epoch = 734, train accuracy = 87.14%, test accuracy = 68.08%
Epoch = 735, train accuracy = 89.01%, test accuracy = 69.74%
Epoch = 736, train accuracy = 90.00%, test accuracy = 69.62%
Epoch = 737, train accuracy = 86.76%, test accuracy = 67.31%
Epoch = 738, train accuracy = 88.19%, test accuracy = 67.69%
Epoch = 739, train accuracy = 87.91%, test accuracy = 68.08%
Epoch = 740, train accuracy = 88.63%, test accuracy = 69.36%
Epoch = 741, train accuracy = 85.77%, test accuracy = 69.36%
Epoch = 742, train accuracy = 83.57%, test accuracy = 67.69%
Epoch = 743, train accuracy = 88.74%, test accuracy = 68.97%
Epoch = 744, train accuracy = 89.40%, test accuracy = 69.36%
Epoch = 745, train accuracy = 85.60%, test accuracy = 68.59%
Epoch = 746, train accuracy = 90.44%, test accuracy = 68.21%
Epoch = 747, train accuracy = 89.89%, test accuracy = 68.85%
Epoch = 748, train accuracy = 88.46%, test accuracy = 68.85%
Epoch = 749, train accuracy = 88.96%, test accuracy = 68.97%
Epoch = 750, train accuracy = 89.12%, test accuracy = 67.95%
Epoch = 751, train accuracy = 90.55%, test accuracy = 68.72%
Epoch = 752, train accuracy = 90.11%, test accuracy = 68.08%
Epoch = 753, train accuracy = 88.13%, test accuracy = 68.85%
Epoch = 754, train accuracy = 89.23%, test accuracy = 68.33%
Epoch = 755, train accuracy = 87.25%, test accuracy = 68.21%
Epoch = 756, train accuracy = 84.01%, test accuracy = 67.95%
Epoch = 757, train accuracy = 82.42%, test accuracy = 68.46%
Epoch = 758, train accuracy = 81.48%, test accuracy = 66.03%
Epoch = 759, train accuracy = 77.86%, test accuracy = 67.44%
Epoch = 760, train accuracy = 83.02%, test accuracy = 66.67%
Epoch = 761, train accuracy = 87.47%, test accuracy = 69.36%
Epoch = 762, train accuracy = 87.47%, test accuracy = 67.56%
Epoch = 763, train accuracy = 88.74%, test accuracy = 66.92%
Epoch = 764, train accuracy = 86.65%, test accuracy = 69.36%
Epoch = 765, train accuracy = 89.01%, test accuracy = 67.69%
Epoch = 766, train accuracy = 89.62%, test accuracy = 68.85%
Epoch = 767, train accuracy = 90.00%, test accuracy = 68.72%
Epoch = 768, train accuracy = 90.38%, test accuracy = 68.72%
Epoch = 769, train accuracy = 87.86%, test accuracy = 68.85%
Epoch = 770, train accuracy = 89.78%, test accuracy = 69.36%
Epoch = 771, train accuracy = 89.56%, test accuracy = 69.87%
Epoch = 772, train accuracy = 87.58%, test accuracy = 68.59%
Epoch = 773, train accuracy = 87.91%, test accuracy = 69.23%
Epoch = 774, train accuracy = 86.65%, test accuracy = 67.56%
Epoch = 775, train accuracy = 90.71%, test accuracy = 69.10%
Epoch = 776, train accuracy = 86.92%, test accuracy = 67.44%
Epoch = 777, train accuracy = 90.33%, test accuracy = 70.90%
Epoch = 778, train accuracy = 87.14%, test accuracy = 67.44%
Epoch = 779, train accuracy = 88.30%, test accuracy = 67.56%
Epoch = 780, train accuracy = 89.01%, test accuracy = 66.92%
Epoch = 781, train accuracy = 88.46%, test accuracy = 68.72%
Epoch = 782, train accuracy = 81.10%, test accuracy = 66.28%
Epoch = 783, train accuracy = 89.40%, test accuracy = 69.10%
Epoch = 784, train accuracy = 88.46%, test accuracy = 69.62%
Epoch = 785, train accuracy = 88.96%, test accuracy = 68.72%
Epoch = 786, train accuracy = 87.86%, test accuracy = 66.92%
Epoch = 787, train accuracy = 83.90%, test accuracy = 67.31%
Epoch = 788, train accuracy = 87.80%, test accuracy = 70.13%
Epoch = 789, train accuracy = 88.74%, test accuracy = 68.85%
Epoch = 790, train accuracy = 89.67%, test accuracy = 68.33%
Epoch = 791, train accuracy = 91.21%, test accuracy = 70.64%
Epoch = 792, train accuracy = 85.55%, test accuracy = 68.59%
Epoch = 793, train accuracy = 89.23%, test accuracy = 69.62%
Epoch = 794, train accuracy = 88.24%, test accuracy = 67.44%
Epoch = 795, train accuracy = 89.40%, test accuracy = 69.10%
Epoch = 796, train accuracy = 89.73%, test accuracy = 67.05%
Epoch = 797, train accuracy = 86.87%, test accuracy = 67.69%
Epoch = 798, train accuracy = 77.53%, test accuracy = 63.85%
Epoch = 799, train accuracy = 88.85%, test accuracy = 67.95%
Epoch = 800, train accuracy = 88.68%, test accuracy = 68.33%
Epoch = 801, train accuracy = 89.67%, test accuracy = 68.46%
Epoch = 802, train accuracy = 90.33%, test accuracy = 67.69%
Epoch = 803, train accuracy = 90.44%, test accuracy = 69.36%
Epoch = 804, train accuracy = 88.30%, test accuracy = 68.97%
Epoch = 805, train accuracy = 88.19%, test accuracy = 68.46%
Epoch = 806, train accuracy = 89.62%, test accuracy = 68.08%
Epoch = 807, train accuracy = 90.00%, test accuracy = 68.59%
Epoch = 808, train accuracy = 85.33%, test accuracy = 66.92%
Epoch = 809, train accuracy = 86.26%, test accuracy = 65.90%
Epoch = 810, train accuracy = 89.84%, test accuracy = 68.46%
Epoch = 811, train accuracy = 88.68%, test accuracy = 67.56%
Epoch = 812, train accuracy = 89.07%, test accuracy = 68.33%
Epoch = 813, train accuracy = 89.62%, test accuracy = 69.87%
Epoch = 814, train accuracy = 89.51%, test accuracy = 68.21%
Epoch = 815, train accuracy = 89.23%, test accuracy = 67.05%
Epoch = 816, train accuracy = 90.22%, test accuracy = 68.85%
Epoch = 817, train accuracy = 90.27%, test accuracy = 69.62%
Epoch = 818, train accuracy = 90.16%, test accuracy = 69.23%
Epoch = 819, train accuracy = 83.08%, test accuracy = 67.31%
Epoch = 820, train accuracy = 86.87%, test accuracy = 68.72%
Epoch = 821, train accuracy = 89.67%, test accuracy = 69.23%
Epoch = 822, train accuracy = 86.10%, test accuracy = 68.97%
Epoch = 823, train accuracy = 83.02%, test accuracy = 67.44%
Epoch = 824, train accuracy = 86.32%, test accuracy = 68.46%
Epoch = 825, train accuracy = 90.27%, test accuracy = 69.10%
Epoch = 826, train accuracy = 89.62%, test accuracy = 68.72%
Epoch = 827, train accuracy = 87.53%, test accuracy = 67.69%
Epoch = 828, train accuracy = 89.84%, test accuracy = 68.59%
Epoch = 829, train accuracy = 87.53%, test accuracy = 67.82%
Epoch = 830, train accuracy = 87.80%, test accuracy = 67.95%
Epoch = 831, train accuracy = 89.62%, test accuracy = 69.10%
Epoch = 832, train accuracy = 89.78%, test accuracy = 69.23%
Epoch = 833, train accuracy = 89.67%, test accuracy = 67.69%
Epoch = 834, train accuracy = 90.11%, test accuracy = 68.21%
Epoch = 835, train accuracy = 87.91%, test accuracy = 67.95%
Epoch = 836, train accuracy = 89.67%, test accuracy = 68.59%
Epoch = 837, train accuracy = 90.60%, test accuracy = 68.21%
Epoch = 838, train accuracy = 86.76%, test accuracy = 68.97%
Epoch = 839, train accuracy = 86.43%, test accuracy = 67.05%
Epoch = 840, train accuracy = 89.95%, test accuracy = 68.08%
Epoch = 841, train accuracy = 89.34%, test accuracy = 70.38%
Epoch = 842, train accuracy = 89.23%, test accuracy = 69.74%
Epoch = 843, train accuracy = 91.92%, test accuracy = 70.51%
Epoch = 844, train accuracy = 90.05%, test accuracy = 67.56%
Epoch = 845, train accuracy = 89.73%, test accuracy = 68.59%
Epoch = 846, train accuracy = 89.67%, test accuracy = 69.87%
Epoch = 847, train accuracy = 85.38%, test accuracy = 67.18%
Epoch = 848, train accuracy = 90.49%, test accuracy = 69.74%
Epoch = 849, train accuracy = 90.77%, test accuracy = 68.97%
Epoch = 850, train accuracy = 89.18%, test accuracy = 66.41%
Epoch = 851, train accuracy = 89.45%, test accuracy = 68.97%
Epoch = 852, train accuracy = 89.95%, test accuracy = 69.87%
Epoch = 853, train accuracy = 91.26%, test accuracy = 69.49%
Epoch = 854, train accuracy = 92.14%, test accuracy = 68.72%
Epoch = 855, train accuracy = 91.32%, test accuracy = 69.36%
Epoch = 856, train accuracy = 89.95%, test accuracy = 68.72%
Epoch = 857, train accuracy = 91.32%, test accuracy = 68.85%
Epoch = 858, train accuracy = 90.38%, test accuracy = 68.46%
Epoch = 859, train accuracy = 91.04%, test accuracy = 68.21%
Epoch = 860, train accuracy = 88.96%, test accuracy = 69.10%
Epoch = 861, train accuracy = 90.22%, test accuracy = 67.95%
Epoch = 862, train accuracy = 81.21%, test accuracy = 66.79%
Epoch = 863, train accuracy = 78.63%, test accuracy = 64.49%
Epoch = 864, train accuracy = 83.30%, test accuracy = 65.13%
Epoch = 865, train accuracy = 90.00%, test accuracy = 70.00%
Epoch = 866, train accuracy = 89.29%, test accuracy = 68.08%
Epoch = 867, train accuracy = 89.12%, test accuracy = 67.56%
Epoch = 868, train accuracy = 91.70%, test accuracy = 68.97%
Epoch = 869, train accuracy = 90.77%, test accuracy = 70.38%
Epoch = 870, train accuracy = 90.77%, test accuracy = 68.59%
Epoch = 871, train accuracy = 91.54%, test accuracy = 68.85%
Epoch = 872, train accuracy = 91.04%, test accuracy = 68.33%
Epoch = 873, train accuracy = 91.15%, test accuracy = 68.59%
Epoch = 874, train accuracy = 90.05%, test accuracy = 68.59%
Epoch = 875, train accuracy = 92.69%, test accuracy = 67.95%
Epoch = 876, train accuracy = 92.09%, test accuracy = 68.97%
Epoch = 877, train accuracy = 91.92%, test accuracy = 68.97%
Epoch = 878, train accuracy = 90.38%, test accuracy = 68.72%
Epoch = 879, train accuracy = 91.37%, test accuracy = 68.59%
Epoch = 880, train accuracy = 87.86%, test accuracy = 67.56%
Epoch = 881, train accuracy = 92.91%, test accuracy = 68.08%
Epoch = 882, train accuracy = 90.71%, test accuracy = 67.69%
Epoch = 883, train accuracy = 93.41%, test accuracy = 69.49%
Epoch = 884, train accuracy = 92.42%, test accuracy = 69.62%
Epoch = 885, train accuracy = 92.69%, test accuracy = 69.23%
Epoch = 886, train accuracy = 92.80%, test accuracy = 68.46%
Epoch = 887, train accuracy = 92.31%, test accuracy = 69.10%
Epoch = 888, train accuracy = 91.15%, test accuracy = 68.59%
Epoch = 889, train accuracy = 92.69%, test accuracy = 68.72%
Epoch = 890, train accuracy = 91.26%, test accuracy = 69.74%
Epoch = 891, train accuracy = 91.26%, test accuracy = 68.59%
Epoch = 892, train accuracy = 89.45%, test accuracy = 69.23%
Epoch = 893, train accuracy = 88.68%, test accuracy = 66.54%
Epoch = 894, train accuracy = 78.52%, test accuracy = 64.23%
Epoch = 895, train accuracy = 84.95%, test accuracy = 66.15%
Epoch = 896, train accuracy = 86.92%, test accuracy = 66.54%
Epoch = 897, train accuracy = 84.89%, test accuracy = 68.08%
Epoch = 898, train accuracy = 79.62%, test accuracy = 67.31%
Epoch = 899, train accuracy = 89.07%, test accuracy = 67.18%
Epoch = 900, train accuracy = 88.46%, test accuracy = 66.03%
Epoch = 901, train accuracy = 92.20%, test accuracy = 69.36%
Epoch = 902, train accuracy = 92.53%, test accuracy = 69.36%
Epoch = 903, train accuracy = 90.55%, test accuracy = 68.59%
Epoch = 904, train accuracy = 92.91%, test accuracy = 69.36%
Epoch = 905, train accuracy = 91.98%, test accuracy = 67.82%
Epoch = 906, train accuracy = 88.79%, test accuracy = 68.46%
Epoch = 907, train accuracy = 93.08%, test accuracy = 69.10%
Epoch = 908, train accuracy = 81.70%, test accuracy = 65.13%
Epoch = 909, train accuracy = 85.38%, test accuracy = 66.03%
Epoch = 910, train accuracy = 87.31%, test accuracy = 67.95%
Epoch = 911, train accuracy = 92.69%, test accuracy = 70.51%
Epoch = 912, train accuracy = 91.92%, test accuracy = 68.33%
Epoch = 913, train accuracy = 93.24%, test accuracy = 69.23%
Epoch = 914, train accuracy = 91.76%, test accuracy = 68.33%
Epoch = 915, train accuracy = 91.26%, test accuracy = 68.72%
Epoch = 916, train accuracy = 91.65%, test accuracy = 66.92%
Epoch = 917, train accuracy = 93.02%, test accuracy = 67.69%
Epoch = 918, train accuracy = 94.07%, test accuracy = 69.10%
Epoch = 919, train accuracy = 93.41%, test accuracy = 67.56%
Epoch = 920, train accuracy = 92.80%, test accuracy = 69.74%
Epoch = 921, train accuracy = 88.08%, test accuracy = 67.56%
Epoch = 922, train accuracy = 92.03%, test accuracy = 68.97%
Epoch = 923, train accuracy = 94.07%, test accuracy = 70.00%
Epoch = 924, train accuracy = 77.69%, test accuracy = 66.03%
Epoch = 925, train accuracy = 82.91%, test accuracy = 66.79%
Epoch = 926, train accuracy = 84.29%, test accuracy = 66.67%
Epoch = 927, train accuracy = 89.45%, test accuracy = 67.69%
Epoch = 928, train accuracy = 85.22%, test accuracy = 67.69%
Epoch = 929, train accuracy = 91.70%, test accuracy = 70.13%
Epoch = 930, train accuracy = 91.10%, test accuracy = 67.69%
Epoch = 931, train accuracy = 92.86%, test accuracy = 68.59%
Epoch = 932, train accuracy = 91.65%, test accuracy = 68.59%
Epoch = 933, train accuracy = 92.80%, test accuracy = 68.97%
Epoch = 934, train accuracy = 92.36%, test accuracy = 67.95%
Epoch = 935, train accuracy = 92.47%, test accuracy = 68.72%
Epoch = 936, train accuracy = 91.21%, test accuracy = 68.33%
Epoch = 937, train accuracy = 92.31%, test accuracy = 69.10%
Epoch = 938, train accuracy = 93.57%, test accuracy = 69.87%
Epoch = 939, train accuracy = 93.19%, test accuracy = 70.51%
Epoch = 940, train accuracy = 94.07%, test accuracy = 70.13%
Epoch = 941, train accuracy = 94.12%, test accuracy = 69.23%
Epoch = 942, train accuracy = 94.07%, test accuracy = 69.10%
Epoch = 943, train accuracy = 94.51%, test accuracy = 69.62%
Epoch = 944, train accuracy = 94.07%, test accuracy = 70.13%
Epoch = 945, train accuracy = 91.26%, test accuracy = 68.59%
Epoch = 946, train accuracy = 77.97%, test accuracy = 63.59%
Epoch = 947, train accuracy = 91.26%, test accuracy = 70.26%
Epoch = 948, train accuracy = 93.19%, test accuracy = 68.72%
Epoch = 949, train accuracy = 84.29%, test accuracy = 65.38%
Epoch = 950, train accuracy = 87.86%, test accuracy = 68.72%
Epoch = 951, train accuracy = 88.57%, test accuracy = 66.54%
Epoch = 952, train accuracy = 91.70%, test accuracy = 69.36%
Epoch = 953, train accuracy = 94.34%, test accuracy = 68.33%
Epoch = 954, train accuracy = 93.24%, test accuracy = 70.00%
Epoch = 955, train accuracy = 94.84%, test accuracy = 68.72%
Epoch = 956, train accuracy = 94.01%, test accuracy = 68.85%
Epoch = 957, train accuracy = 85.44%, test accuracy = 65.13%
Epoch = 958, train accuracy = 89.95%, test accuracy = 68.33%
Epoch = 959, train accuracy = 89.51%, test accuracy = 70.13%
Epoch = 960, train accuracy = 92.25%, test accuracy = 68.72%
Epoch = 961, train accuracy = 92.53%, test accuracy = 68.33%
Epoch = 962, train accuracy = 94.56%, test accuracy = 68.72%
Epoch = 963, train accuracy = 93.90%, test accuracy = 68.59%
Epoch = 964, train accuracy = 92.80%, test accuracy = 69.10%
Epoch = 965, train accuracy = 93.90%, test accuracy = 68.46%
Epoch = 966, train accuracy = 93.24%, test accuracy = 68.72%
Epoch = 967, train accuracy = 95.22%, test accuracy = 69.36%
Epoch = 968, train accuracy = 93.13%, test accuracy = 70.51%
Epoch = 969, train accuracy = 94.29%, test accuracy = 69.23%
Epoch = 970, train accuracy = 93.57%, test accuracy = 68.21%
Epoch = 971, train accuracy = 94.73%, test accuracy = 69.23%
Epoch = 972, train accuracy = 95.44%, test accuracy = 70.00%
Epoch = 973, train accuracy = 93.35%, test accuracy = 68.33%
Epoch = 974, train accuracy = 94.89%, test accuracy = 68.97%
Epoch = 975, train accuracy = 92.97%, test accuracy = 69.10%
Epoch = 976, train accuracy = 95.38%, test accuracy = 68.72%
Epoch = 977, train accuracy = 89.18%, test accuracy = 68.72%
Epoch = 978, train accuracy = 87.69%, test accuracy = 65.38%
Epoch = 979, train accuracy = 92.31%, test accuracy = 69.49%
Epoch = 980, train accuracy = 92.53%, test accuracy = 68.85%
Epoch = 981, train accuracy = 94.40%, test accuracy = 66.28%
Epoch = 982, train accuracy = 92.14%, test accuracy = 68.59%
Epoch = 983, train accuracy = 94.73%, test accuracy = 68.85%
Epoch = 984, train accuracy = 92.91%, test accuracy = 69.23%
Epoch = 985, train accuracy = 91.32%, test accuracy = 68.97%
Epoch = 986, train accuracy = 83.08%, test accuracy = 66.67%
Epoch = 987, train accuracy = 86.92%, test accuracy = 65.51%
Epoch = 988, train accuracy = 93.02%, test accuracy = 68.85%
Epoch = 989, train accuracy = 94.01%, test accuracy = 68.85%
Epoch = 990, train accuracy = 94.89%, test accuracy = 69.87%
Epoch = 991, train accuracy = 94.78%, test accuracy = 67.69%
Epoch = 992, train accuracy = 94.84%, test accuracy = 68.46%
Epoch = 993, train accuracy = 94.78%, test accuracy = 68.46%
Epoch = 994, train accuracy = 95.11%, test accuracy = 69.10%
Epoch = 995, train accuracy = 95.05%, test accuracy = 69.87%
Epoch = 996, train accuracy = 93.74%, test accuracy = 68.85%
Epoch = 997, train accuracy = 95.77%, test accuracy = 68.97%
Epoch = 998, train accuracy = 95.82%, test accuracy = 69.74%
Epoch = 999, train accuracy = 95.22%, test accuracy = 68.85%
Epoch = 1000, train accuracy = 95.71%, test accuracy = 69.74%
Time taken: 0:15:24.420289

Process finished with exit code 0


'''
