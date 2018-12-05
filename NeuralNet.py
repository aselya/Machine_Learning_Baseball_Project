import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
print("test of imports")
#http://eerwitt.github.io/2016/01/14/tensorflow-from-csv-to-api/
file = "/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv"
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(file),shuffle=True)

# Each file will have a header, we skip it and give defaults and type information
# for each column below.
line_reader = tf.TextLineReader(skip_header_lines=1)

_, csv_row = line_reader.read(filename_queue)

# Type information and column names based on the decoded CSV.
record_defaults = [[0.0],[0.0], [0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0], [0.0],['']]
R,AB,H, Double, Triple,HR,BB,SO,SB,CS,HBP,SF,RA,ER,ERA,CG,SHO,SV,IPouts,HA,HRA,BBA,SOA,E,DP,FP,Classification = \
    tf.decode_csv(csv_row, record_defaults=record_defaults)

features = tf.stack([
    R,AB,H, Double, Triple,HR,BB,SO,SB,CS,HBP,SF,RA,ER,ERA,CG,SHO,SV,IPouts,HA,HRA,BBA,SOA,E,DP,FP])

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

     # We do 10 iterations (steps) where we grab an example from the CSV file.
    for iteration in range(1, 11):
        # Our graph isn't evaluated until we use run unless we're in an interactive session.
        example, label = sess.run([features, Classification])

        print(example, label)
    coord.request_stop()
    coord.join(threads)

all_Catagories_of_teams = ["Bad Team", "Good Team", "Average Team", "Playoff Team"]
onehot = {}
# Target number of species types (target classes) is 3 ^
categories_count = len(all_Catagories_of_teams)

# Print out each one-hot encoded string for 3 species.
for i, categories in enumerate(all_Catagories_of_teams):
    # %0*d gives us the second parameter's number of spaces as padding.
    print("%s,%0*d" % (species, categories_count, 10 ** i))

