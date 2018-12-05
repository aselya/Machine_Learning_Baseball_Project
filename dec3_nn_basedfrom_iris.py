import pandas as pd
import tensorflow as tf


from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from sklearn.model_selection import train_test_split

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    
    #fetch data
    csv_Dataframe = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv")
    csv_Dataframe = csv_Dataframe.fillna(1) #replaces all NAN with 1 representing average
    #one hot encoding
    dataframe_lb = LabelBinarizer()
    Y = dataframe_lb.fit_transform(csv_Dataframe.Classification.values)
    print("Y" + str(Y))


    #eliminates some more stats
    csv_Dataframe_reduced = dataframe.drop(['DP','E', 'IPouts', 'E', 'CG', 'SF', 'HBP'], axis=1)




    features_max_index = len(csv_Dataframe_reduced.columns) - 1
    FEATURES = csv_Dataframe_reduced.columns[0:features_max_index]
    print("features: "+ str(FEATURES))
    X_data = csv_Dataframe_reduced[FEATURES].as_matrix()
    X_data = normalize(X_data)


    X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=1)
    X_train.shape
    print("X_train shape" + str(X_train.shape)) #1820, 26| features
    print("Y_train shape" + str(y_train.shape)) #1820, 4 | 4 -> bad tema, average team, good team, playoff team

    
    my_feature_columns = []
    for key in X_train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        
    classifer = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units = [10,10],
        n_classes= 4
        
    )    
    
    classifer.train(
        
        input_fn=lambda : 
        
    )





