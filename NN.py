'''

Leauge ave = 81 wins per season = 50% winning percentage = 1 on ave_stat_by_team.csv
Bad team: < 77            | 77/162 = .47530864   | 77/81 = .95061728
Average Team :77 < x < 85 | 84/162 = .51851852   | 84/81 = 1.03703704
Good Team: 85-92          | 92/162 = .56790123   | 92/81 = 1.13580247
Playoff team: > 92 wins   | 


'''



import matplotlib as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

print('test to make sure imports were accepted')

def read_dataset():
    df = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_col.csv")
    #start x at index 4 since the first 4 stats aren't relevant
    #year id will always be 1, # of games doesn't impact playoffs because it's done on %
    #wins and loss data is used in the classification section
    X = df[df.columns[4:30]]
    y = df[df.columns[30]]


#https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
    #encode the dependent vars
    encoder = LabelEncoder()
    encoder.fit_transform(y)
    #encoder.fit(y)
    y = encoder.transform(y)

    Y = one_hot_encode(y)
    Y = float(Y)
    print(X.shape)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros(n_labels, n_labels)
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

#x is the features and Y is the Labels
X,Y = read_dataset()

#Shuffle the data to mix up the rowa
X, Y = shuffle(X,Y, random_state=1)

#convert dataset into train and test data
#where test data is 20% of the size
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.20, random_state=415)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)


#parameters of NN

learning_rate = 0.3
trainingPepochs =  1000
cost_history =  np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
ptint("n_dim", n_dim)
n_class = 4 #bad team, average, good, playoff
model_path = "/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/NN_model"

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros(n_class))
y_ = tf.placeholder(tf.float32, [None, n_class])


