import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from datetime import datetime

print("imports worked")

dataset = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv', index_col=0)
print(dataset.head(10))
dataset.head()

team_quality = LabelBinarizer()
Y = team_quality.fit_transform(dataset.Classification.values)
print('binary classification complete')
print(Y)

FEATURES = dataset.columns[0:25]

dataset.fillna(1, inplace=True)
print('replaced all na with 1 representing leauge average')

X_data = dataset[FEATURES].as_matrix()
X_data = normalize(X_data)
print('normalizing X_data complete')

X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.2, random_state=1)
X_train.shape

print("x_train shape:" + str(X_train.shape))
print("X_test shape: "+ str(X_test.shape))
print("y_train shape:" + str(y_train.shape))
print("y_test shape: "+ str(y_test.shape))


#parameters
learning_rate = 0.01
training_epochs = 200

#Neural network paramerters
n_hidden_1 = 25 #first hidden layer
n_hidden_2 = 25 #second hidden layer

n_input = X_train.shape[1]
#n_input = tf.reshape(n_input , [-1 , 25])


print('n_input: ' +str(n_input))

n_classes = y_train.shape[1]
print('n_classes: ' +str(n_classes))



#inputs

X = tf.placeholder("float", shape=[None, n_input])
#y = tf.placeholder("float", shape=[None, n_input])
y = tf.placeholder('float')


weights = {
    'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def forward_propagation(x):
    #hidden layer 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(x, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    #output fully connected layer
    out_layer =  tf.matmul(layer_2, weights['out'])+ biases['out']
    return out_layer

yhat = forward_propagation(X)
ypredicat = tf.argmax(yhat, axis=1)



#backwards prop

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train_op = optimizer.minimize(cost)

#training time
init = tf.global_variables_initializer()

startTime = datetime.now()

with tf.Session() as sess:
    sess.run(init)


    #writer.add_graph
    #epochs
    for epoch in range(training_epochs):
        #stochasting gradient descent

        for i in range(len(X_train)):
            summary =  sess.run(train_op, feed_dict={X: X_train[i: i+1], y: y_train[i: i+1]})

            #summary =  sess.run(train_op, feed_dict={X: X_train[i: i+1], y: y_train[i: i+1]})

        train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredicat, feed_dict={X: X_train, y: y_train}))
        print("print train accuracy shape"+str(train_accuracy))
        #train_accuracy = tf.reshape(train_accuracy, None) #TypeError: only size-1 arrays can be converted to Python scalars


        test_accuracy = np.mean(np.argmax(y_test, axis=1)) == sess.run(ypredicat, feed_dict={X: X_test, y: y_test})
        print("print test accuracy shape"+str(test_accuracy) + "end shape")

        #test_accuracy = tf.reshape(test_accuracy, [1, None])

        print("Epoch = %d, Train accuracy = %.2f%%, test accuracy = %.2f%%" %(epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

print("time taken: ", datetime.now()-startTime)





