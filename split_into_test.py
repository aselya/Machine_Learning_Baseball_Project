import pandas as pd
from sklearn.model_selection import train_test_split
import random

file = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/stats_like_iris.csv")

test_rate = .25

#read rows
#if row < probability write to test
#if not write to file
test = ''
data = ''

for line in file:
    rand = 0
    rand = random.random()
    if rand < test_rate:
        test = test + line + '/n'
    else:
        data = data + line + '/n'





print(test)
print(data)
