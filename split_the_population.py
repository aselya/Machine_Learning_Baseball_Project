#/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/stats_like_iris.csv

from sklearn.model_selection import _split

import pandas as pd

data = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv")
data_noNA = data.fillna(1)

pop1 , pop2 = _split.train_test_split(data_noNA)

print(pop1)
print(type(pop1))
print(pop2)
print(type(pop2))

pop1.to_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/csv_data.csv', index=False)
pop2.to_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/csv_test.csv', index=False)






