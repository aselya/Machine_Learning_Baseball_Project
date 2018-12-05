#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


#parser = argparse.ArgumentParser()
#parser.add_argument('--batch_size', default=100, type=int, help='batch size')
#parser.add_argument('--train_steps', default=1000, type=int,
#                    help='number of training steps')
BATCH_SIZE = 300
TRAIN_STEPS = 3000

tf.logging.set_verbosity(tf.logging.INFO)


def make_neural_net(my_feature_columns):
    neural_net =tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[50, 50, 50],
        # The model must choose between 3 classes.
        n_classes=4)
    return neural_net

def get_the_data():
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    return (train_x, train_y), (test_x, test_y)

def set_feature_columns( train_x):
        my_feature_columns = []
        for key in train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        print("my_feature_columns" + str(my_feature_columns))
        return my_feature_columns

def train_the_model( classifier, train_x, train_y, ):
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 BATCH_SIZE),
        steps=TRAIN_STEPS)

def evaluate_the_results(classifier, test_x, test_y):
    eval_result = classifier.evaluate(input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,BATCH_SIZE))
    return eval_result


def get_prediction(classifier, input_dictionary):

    predictions = classifier.predict(input_fn=lambda:iris_data.eval_input_fn(input_dictionary,
                                labels=None,batch_size=BATCH_SIZE))
    template = ('\nPrediction is "{}" ({:.1f}%), ')


    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        actual_prediction = iris_data.SPECIES[class_id]
        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability))
    return actual_prediction

#def main(argv):
    #args = parser.parse_args(argv[1:])
    #print("args" + str(args))
    # Fetch the data
    #(train_x, train_y), (test_x, test_y) = iris_data.load_data()

(train_x, train_y), (test_x, test_y)= get_the_data()
    #print(type(train_x))
    #print(train_x.shape)
print("iris_data.load" + str(train_x) + str(train_y))
print("iris_data.load" + str(test_x) + str(test_y))

 # Feature columns describe how to use the input.
'''
my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    print("my_feature_columns" + str(my_feature_columns))
    '''
my_feature_columns = set_feature_columns( train_x)
    # Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = make_neural_net(my_feature_columns)


print("train x and y shape" +str(train_x.shape) )
print(str(train_y.shape))
#print(str(args.batch_size))

train_the_model( classifier, train_x, train_y, )
# Train the Model.
'''
    classifier.train(

        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)
    '''
    # Evaluate the model.
eval_result = evaluate_the_results(classifier, test_x, test_y)
'''
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))
    '''
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    #expected = ['Bad Team', 'Average Team', 'Good Team', 'Playoff Team']
predict_x = {
       'R': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'AB': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'H': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        '2B': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        '3B': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HR': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'BB': [1 , 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SO': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SB': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'CS': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HBP': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SF': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'RA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'ER': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'ERA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'CG': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SHO': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SV': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'IPouts': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HRA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'BBA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SOA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'E' : [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'DP': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'FP': [1, 1.2, 1.222 , 1.11111111 , 1.11111111]



       # 'SepalLength': [5.1, 5.9, 6.9],
        #'SepalWidth': [3.3, 3.0, 3.1],
        #'PetalLength': [1.7, 4.2, 5.4],
        #'PetalWidth': [0.5, 1.5, 2.1],
    }

predict_x2 = {'BB': 1.009641, 'SOA': 1.004817, 'RA': 0.969584, 'DP': 0.991801, 'AB': 0.967161, 'SO': 1.030452, 'SB': 1.032536, 'SV': 0.966212, '3B': 1.005646, 'HRA': 0.960336, 'ERA': 1.04304, 'SHO': 1.030958, '2B': 0.9806010000000001, 'R': 0.964825, 'ER': 0.991213, 'FP': 1.03497, 'HBP': 0.967475, 'IPouts': 0.971961, 'CG': 0.997358, 'E': 1.035217, 'HA': 0.973221, 'H': 1.009223, 'CS': 0.981622, 'SF': 0.968128, 'HR': 0.992999, 'BBA': 1.044765}
predict_x3={'ERA': [1.021266], 'SO': [0.989227], 'BBA': [0.970416], 'RA': [0.967489], 'BB': [1.041685], 'SF': [1.024911], 'E': [0.956521], 'HR': [1.043219], 'ER': [0.9599530000000001], 'HA': [0.995577], 'R': [0.970591], 'SHO': [1.008279], 'HRA': [0.981935], 'CG': [1.000704], 'CS': [1.021651], 'AB': [0.969144], 'SV': [1.01259], 'SOA': [1.044072], 'IPouts': [1.041474], 'H': [1.015401], '3B': [0.966425], 'FP': [0.995196], 'HBP': [0.989337], 'SB': [1.015562], '2B': [1.021475], 'DP': [1.028915]}


print(type(predict_x3))

predictions = get_prediction(classifier, predict_x3)


'''
    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x3,
                                                labels=None,
                                                batch_size=args.batch_size))
    '''

#template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

'''
    template = ('\nPrediction is "{}" ({:.1f}%), ')


    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability))
'''
'''
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))
'''

#if __name__ == '__main__':
#print(tf.logging.set_verbosity(tf.logging.INFO))
  #  tf.app.run(main)
