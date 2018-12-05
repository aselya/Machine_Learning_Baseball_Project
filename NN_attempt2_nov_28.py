import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
#kaggle ellinox tensorflow neural network tutorial with iris

csv_Dataframe = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv")

print(csv_Dataframe.head(10))

print("csv_Dataframe shape: "+ str(csv_Dataframe.shape))

print("csv_Dataframe dataTypes: "+ str(csv_Dataframe.dtypes))

#convert the classification to ints
csv_Dataframe["Classification"].map({"Bad Team":0, "Average Team":1, "Good Team":2, "Playoff Team":3})

print(csv_Dataframe.head())

X_train, X_test, y_train, y_test = train_test_split(csv_Dataframe.iloc[:,1:26], csv_Dataframe["Classification"], test_size = 0.2, random_state = 42)

X_train.shape
print("X_train shape" + str(X_train.shape))

columns = csv_Dataframe.columns[:26]
print("columns:" +str(columns) )

learning_rate = 0.01
learning_epochs = 100

n_hidden_1 = 256
n_hidden_2 = 128
n_input = X_train.shape[1]
print("x_train: " +str(X_train.shape[1]))
n_classes = y_train.shape
print("y_train: " +str(y_train.shape)[1])



'''
feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]

def input_fn(df, labels):
    feature_columns = {k: tf.constant(df[k].values,shape = [df[k].size, 1]) for k in columns}
    label = tf.constant(labels.values, shape= [labels.size, 1])
    return feature_columns, label

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10,20,10],n_classes = 4)

classifier.fit(input_fn=lambda: input_fn(X_train, y_train), steps = 100)
'''
