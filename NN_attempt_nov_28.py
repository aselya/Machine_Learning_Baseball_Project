#attempt  tensorflow basics deeplearning wuth neural nets p. 3
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd

#gets the csv
csv_dataframe = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv")

#converts the classification column to int values
print(csv_dataframe.head(10))
csv_dataframe = csv_dataframe['Classification'].replace(['Bad Team', 'Average Team', 'Good Team', 'Playoff Team'], [0, 1, 2, 4])
print(csv_dataframe.head(10))




#df[2].replace(4, 17)
" /Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv"
"R,AB,H,2B,3B,HR,BB,SO,SB,CS,HBP,SF,RA,ER,ERA,CG,SHO,SV,IPouts,HA,HRA,BBA,SOA,E,DP,FP,Classification"
"26 catagories and one label"




#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

tf.enable_eager_execution()

features = ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP']

print("print features values" + [features].values)
training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        [
            tf.cast(csv_dataframe[features].values, tf.float32),
            tf.cast(csv_dataframe['Classification'].values, tf.int32)
        ]
   )
)

for features_tensor, target_tensor in training_dataset:
    print('features:{features_tensor} target:{target_tensor}')

'''
 input > weight > hidden layer 1 (activation function) > weights > hidden L2
 (activation function) > weisghts > output layer


compare output to intended  output > cost or loss function

optimization functuon (opimizer )> minimize cost (adamoptimized) 

back propigation

feed forward + backprop = epoch

'''


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 4 # represents the number of possible classifications

batch_size = 100 # breaks data up in to smaller batched

#height by width
#  x input data
x = tf.placeholder('float', [None, 26])  # change to values of columns
    #good to leave to make sure that the input is same shape
y = tf.placeholder('float')


def neural_network_model(data): #takes in data sets up computation graph
    #dictionaries
    #shape is number of input vs nodes in hidden layer i  @17:33
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([26, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))} # no need for a shape
                        #biases are added at the end after evereything comes through to avoid no neurons firising if input data is 0
                        #input * weights  + bias
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))} #output is # of classes

    #input * weights  + bias

    #adds the value of the matrix multiplication of data and hidden layer  weights to the biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
    #relu is the activation function (recified liniar)
    li = tf.nn.relu(l1)

    #L! activation value gets passed to layer 2
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    #no relu on output
    output = tf.add(tf.matmul(l3, output_layer['weights']) + output_layer['biases'])

    return output

def train_neural_network(x):
        prediction = neural_network_model(x)
        #cross enthropy with logits is the cost function
        #compares the prediction to the actual results
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y))

        #socrastic gradient descent
        #learning rate is 0.001 by default
        opimizer = tf.train.AdamOptimizer().minimize(cost)

        #cycles of feed forward + back pro to fix all weights
        hm_epochs = 10

        with tf.session() as sess: #begins session
            sess.run(tf.initialize_all_variables())

            #the for loop trains the network
            for epoch in range(hm_epochs):
                epoch_loss = 0

                #takes advantage of prebuilt batchsize
                for _ in range(int(mnist.train.num_examples/batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    #optimize the costs with X's and Y's
                    #optimizing by tensorflow modifing weights
                    _, c = sess.run([opimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                    epoch_loss += c
                print('Epoch', epoch, ' completed out of', hm_epochs, 'loss: ', epoch_loss)

            #returns index value of maxiumum in array by compating tells us if identical
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            #evaluate test data to its labels
            print('accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


#train_neural_network(x)
