import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

'''
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
'''

CSV_COLUMN_NAMES =['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','Classification']

#SPECIES = ['Setosa', 'Versicolor', 'Virginica']

SPECIES = ['Bad Team', 'Good Team', 'Average Team', 'Playoff Team']


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    data = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv', names=CSV_COLUMN_NAMES, header=1)
    data = data.fillna(1) #fill na

    #train_path, test_path = maybe_download()

    #train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    #train_x, train_y = train, train.pop(y_name)

    #test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    #test_x, test_y = test, test.pop(y_name)
    y = data.Classification.values


    features_max_index = len(data.columns) - 1
    FEATURES = data.columns[0:features_max_index]
    print("features: "+ str(FEATURES))
    X_data = data[FEATURES].as_matrix()
    X_data = normalize(X_data)

    
    train_x, test_x, train_y, test_y = train_test_split(X_data, y, test_size=.2, random_state=42)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    print(str(labels))
    print(type(labels))
    print("labels shape" + str(labels.shape))
    print(str(features))
    print(type(features))
    print("features shape" + str(labels.shape))

    #new_col = np.append(features, labels, 1)
    #new_col.shape

    #dataset = tf.data.Dataset.from_tensor_slices((dict(new_col)))

    dataset = tf.data.Dataset.
    #dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0], ['']]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
