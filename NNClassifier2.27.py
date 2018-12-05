import tensorflow as tf
import numpy as np
import pandas
import sys


#
# USAGE: $ python3 csv-to-tfrecords.py data.csv data.tfrecords
#


#infile=sys.argv[1]
#outfile=sys.argv[2]

csv = pandas.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv").values


with tf.python_io.TFRecordWriter('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/NNCLassifier2_27.tfrecords') as writer:
    for row in csv:

        ## READ FROM CSV ##

        # row is read as a single char string of the label and all my floats, so remove trailing whitespace and split
        row = str(row[0]).split(' ')
        # the first col is label, all rest are feats
        label = int(float(row[0]))
        # convert each floating point feature from char to float to bytes
        feats = np.array([ float(feat) for feat in row[1:] ]).tostring()

        ## SAVE TO TFRECORDS ##

        # A tfrecords file is made up of tf.train.Example objects, and each of these
        # tf.train.Examples contains one or more "features"
        # use SequenceExample if you've got sequential data

        example = tf.train.Example()
        example.features.feature["feats"].bytes_list.value.append(feats)
        example.features.feature["label"].int64_list.value.append(label)
        writer.write(example.SerializeToString())




def parser(record):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically,
    this function defines what the labels and data look like
    for your labeled data.
    '''

    # the 'features' here include your normal data feats along
    # with the label for that data
    features={
      'feats': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    }

    parsed = tf.parse_single_example(record, features)

    # some conversion and casting to get from bytes to floats and ints
    feats= tf.convert_to_tensor(tf.decode_raw(parsed['feats'], tf.float64))
    label= tf.cast(parsed['label'], tf.int32)

    # since you can have multiple kinds of feats, you return a dictionary for feats
    # but only an int for the label
    return {'feats': feats}, label


DNNClassifier = tf.estimator.DNNClassifier(

   # for a DNN, this feature_columns object is really just a definition
   # of the input layer
   feature_columns = [tf.feature_column.numeric_column(key='feats',
                                                       shape=(377,),
                                                       dtype=tf.float64)],

   # four hidden layers with 256 nodes in each layer
   hidden_units = [256, 256, 256, 256],

   # number of classes (aka number of nodes on the output layer)
   n_classes = 4,

)

def my_input_fn():
    tfrecords_path ='/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/NNCLassifier2_27.tfrecords'
    dataset = (
      tf.data.TFRecordDataset(tfrecords_path)
     .map(parser)
     .batch(1024)
    )

    iterator = dataset.make_one_shot_iterator()

    batch_feats, batch_labels = iterator.get_next()

    return batch_feats, batch_labels


dataset = (
    tf.data.TFRecordDataset('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/NNCLassifier2_27.tfrecords')
    .map(parser)
    .shuffle(buffer_size=1024)
    .batch(32)
)

DNNClassifier = tf.estimator.DNNClassifier(
  feature_columns = [tf.feature_column.numeric_column(key='feats', dtype=tf.float64, shape=(377,))],
  hidden_units = [256, 256, 256, 256],
  n_classes = 4,
  model_dir = '/tmp/tf')

