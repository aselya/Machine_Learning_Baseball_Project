import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops import rnn, rnn_cell



print("inports worked")

from sklearn.preprocessing import normalize

csv_Dataframe = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv")
csv_Dataframe = csv_Dataframe.fillna(1) #replaces all NAN with 1 representing average
#one hot encoding
dataframe_lb = LabelBinarizer()
Y = dataframe_lb.fit_transform(csv_Dataframe.Classification.values)
print("Y" + str(Y))


#eliminates some more stats
def reduce_cols (dataframe):
    new_df = dataframe.drop(['DP','E', 'IPouts', 'E', 'CG', 'SF', 'HBP'], axis=1)

    print("reduced dataset dataframe:" + str (dataframe.head(1)))
    return new_df

csv_Dataframe_reduced = reduce_cols(csv_Dataframe)




features_max_index = len(csv_Dataframe_reduced.columns) - 1
FEATURES = csv_Dataframe_reduced.columns[0:features_max_index]
print("features: "+ str(FEATURES))
X_data = csv_Dataframe_reduced[FEATURES].as_matrix()
X_data = normalize(X_data)

#print(X_data)

test_dataframe = csv_Dataframe_reduced.head(1)
testing_dataFrame = test_dataframe.columns[0:features_max_index]
testing_data = test_dataframe[FEATURES].as_matrix()
testing_data = normalize(testing_data).astype(np.float32)
print("tesint data:" + str(testing_data.shape))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=1)
X_train.shape
print("X_train shape" + str(X_train.shape)) #1820, 26| features
print("Y_train shape" + str(y_train.shape)) #1820, 4 | 4 -> bad tema, average team, good team, playoff team



# Parameters
learning_rate = 0.05
training_epochs = 10


n_input = X_train.shape[1] # 26
n_classes = y_train.shape[1] # 4


# Inputs
X = tf.placeholder("float", shape=[None, n_input])
y = tf.placeholder("float", shape=[None, n_classes])


#####
hm_epochs = 10
n_classes = 4
batch_size = 20
n_chunks = 40
chunk_size = 40
rnn_size = 128

x = tf.placeholder("float", shape=[None, n_chunks,chunk_size])
y = tf.placeholder("float", shape=[None, n_classes])


n_classes = 4 # represents the number of possible classifications

batch_size = 100 # breaks data up in to smaller batched



def reccurant_neural_network_model(data): #takes in data sets up computation graph
    #dictionaries
    #shape is number of input vs nodes in hidden layer i  @17:33
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))} # no need for a shape
                        #biases are added at the end after evereything comes through to avoid no neurons firising if input data is 0
                        #input * weights  + bias

    data = tf.transpose(data, [1,0,2]) #formats data so tensorflow rnn_cell likes it
    data = tf.reshape(data, [-1, chunk_size])
   # data = tf.split(0, n_chunks, data)

    lstm_cell =  rnn_cell.BasicLSTMCell(rnn_size)
    outputs , states = rnn.rnn(lstm_cell, data, dtype = tf.float32)


    #no relu on output
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    print(output)
    return output

#saver = tf.train.Saver()
#tf_log = 'tf_log'

def train_neural_network(x):
        prediction = reccurant_neural_network_model(x)
        #cross enthropy with logits is the cost function
        #compares the prediction to the actual results
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))

        #socrastic gradient descent
        #learning rate is 0.001 by default
        opimizer = tf.train.AdamOptimizer().minimize(cost)

        #cycles of feed forward + back pro to fix all weights
        hm_epochs = 10

        with tf.Session() as sess: #begins session
            sess.run(tf.initialize_all_variables())

            #the for loop trains the network
            epoch_loss = 0
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(X_data/batch_size):
                    epoch_x, epoch_y = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                    #optimize the costs with X's and Y's
                    #optimizing by tensorflow modifing weights
                    _, c = sess.run([opimizer, cost], feed_dict = {x: X_data, y: Y})
                    epoch_loss += c
                print('Epoch', epoch, ' completed out of', hm_epochs, 'loss: ', epoch_loss)

            #returns index value of maxiumum in array by compating tells us if identical
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
                #print("correct" + str(correct))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            #evaluate test data to its labels
            print('accuracy: ', accuracy.eval({x:X_data.reshape((-1, n_chunks, chunk_size)), y:Y.reshape((-1, n_chunks, chunk_size))}))



#train_neural_network(x)
train_neural_network(x)


#test_dataframe = test_data.as_matrix

string_predictions = neural_network_model(testing_data)
print('predictions: ' + str(string_predictions) )

# 8 deep learning with NN @18
def use_neural_network(input_data):
    prediction = neural_network_model(x)

    features = np.array(testing_data)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        result = sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}), 1))

        print('resutl' + result[0])

use_neural_network(testing_data)
