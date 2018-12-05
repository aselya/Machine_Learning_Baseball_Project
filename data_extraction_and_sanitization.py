import pandas as pd
import csv
import numpy as np
from pandas import DataFrame as df


lahmenCSV = pd.read_csv(r'/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team.csv')

print(lahmenCSV.head(10))
#some stats such as home park are not needed
relevant_Stats = ['yearID','G','W','L','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP']


#make file with the headers of the relevant stats
with open(r'ave_stats_by_year.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(relevant_Stats)


def get_stats_for_year(currentYear):
    df = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/relevant_stats.csv')
    print('get_stats_for_year(currentYear)')
    print( df.head(10))
    current_year_stats = df.loc[df['yearID'] == currentYear]
    print (current_year_stats)
    return current_year_stats



#for each year get the total for each stat and write the average
def get_yearly_averages( startYear, endYear):
    currentYear = startYear
    while currentYear <= endYear:
        get_stats_for_year(currentYear)
        currentYear += 1


#get averages for each stat and add them to the ave_stat_by_year.csv file
def calculate_average(dataFrame):
    average_stats = []
    print("describe method")
    print(dataFrame.describe)
    print(dataFrame.mean(axis=0))
    average_stats = dataFrame.mean(axis=0)

    #for columns in dataFrame:
     #   if columns != "lgID":
      #      average_stats[columns] = int(columns)

    print(average_stats)
    return average_stats

def add_average_stats_to_csv(dataframe):
    dataframe.to_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/ave_stats_per_year.csv')
    #dataframe.to_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/ave_stats_per_year.csv')
    with open(r'ave_stats_by_year.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(dataframe)



def add_average_stats_for_range_of_years_to_csv( start, end):
    count = 0
    while count + start <= end:
        current_year = start+count
        add_average_stats_to_csv(calculate_average(get_stats_for_year(current_year)))
        count += 1


def recall_average_stats_for_a_given_year(currentYear):
    df = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/ave_stats_per_year.csv')
    print('recall_average_stats_for_a_given_year')
    current_year_stats = df.loc[df['yearID'] == currentYear]
    print (current_year_stats)
    return current_year_stats

def recall_team_stats_for_a_given_year(currentYear):
    df = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/relevant_stats.csv')
    print('recall_team_stats_for_a_given_year(currentYear)')
    current_year_team_stats= df.loc[df['yearID'] == currentYear]
    print (current_year_team_stats)
    return current_year_team_stats




def compare_teamStats_to_leauge_Ave( currentYear):

    average = recall_average_stats_for_a_given_year(currentYear)
    teams = recall_team_stats_for_a_given_year(currentYear)
    print('compare_teamStats_to_leauge_Ave( currentYear)')
    average = teams.values / average.values
    print(average)
    averageDataframe = pd.DataFrame(average)
    print(averageDataframe)
    return averageDataframe

#writes the team yearly comparisions to the CSV file
def write_teamStats_to_leauge_ave_to_csv(dataframe):
    dataframe.to_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/ave_stats_by_team.csv')
    with open(r'ave_stats_by_team.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(dataframe)

#isolates team stats for a given range appends to same dataframe
#adds headers to the dataframe and writes results to csv
def add_teamStats_to_leauge_ave_for_range (start , end):
    count = 0
    team_stats = compare_teamStats_to_leauge_Ave(start)
    team_stats.columns = relevant_Stats

    count +=1
    while count + start <= end:
        current_year = start+count
        team = compare_teamStats_to_leauge_Ave(current_year)
        team.columns = relevant_Stats
        #dataframe_with_all_data.append(team, ignore_index=True)
        #team.columns = relevant_Stats
        team_stats = team_stats.append(team, ignore_index=True)
        count += 1
        
    write_teamStats_to_leauge_ave_to_csv(team_stats)
    print(team_stats.head(40))
    return team_stats

def add_classification_column_to_data():
    df = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/ave_stats_by_team.csv')
    determine_classification(df)
    #df["classification"] = np.nan
    print(df.head(10))
    return df


'''

Leauge ave = 81 wins per season = 50% winning percentage = 1 on ave_stat_by_team.csv
Bad team: < 77            | 77/162 = .47530864   | 77/81 = .95061728
Average Team :77 < x < 85 | 84/162 = .51851852   | 84/81 = 1.03703704
Good Team: 85-92          | 92/162 = .56790123   | 92/81 = 1.13580247
Playoff team: > 92 wins   | 


'''
def calculate_classification( row ):
    value = row['W']
    value = float(value)
    if value < float(0.95061728):
        return 'Bad_Team'
    if float(0.95061728) < value < float(1.03703704):
        return 'Average_Team'
    if float(1.03703704) < value < float(1.13580247):
        return 'Good_Team'
    if float(1.13580247) < value:
        return 'Playoff_Team'
    return NaN


def determine_classification(dataFrame):
    df = dataFrame
    df['Classification'] = df.apply (lambda row: calculate_classification( row ),axis=1)
    df.head(15)
    return df

#writes the team yearly comparisions to the CSV file
def write_teamStats_to_leauge_ave_to_csv(dataframe):
    dataframe.to_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/stats_with_classification.csv')
    with open(r'stats_with_classification.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(dataframe)

#transformations added additional unneeded columns to the data this gets rid of them
#games wins and losses are unneeded because of the year average
#year ID is also unneeded because it will be 1 for all teams
def remove_excess_columns_classification_column_to_data():
    df = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/stats_with_classification.csv')
    #adding inplace allows us to drop the excess columns without reassigning dataframe
    df.drop(columns=['added', 'added2', 'yearID','G','W','L'], axis=1, inplace=True)
    print(df.head(10))
    return df

def write__excess_columns_removed_teamStats_to_leauge_ave_to_csv(dataframe):
    #adding index=false removes the issue with additional column from earlier
    dataframe.to_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv', index=False)
    with open(r'stats_with_classification.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(dataframe)



#relevant_Stats_file_lahmenCSV = lahmenCSV[relevant_Stats]

#print(relevant_Stats_file_lahmenCSV.head(10))

#relevant_Stats_file_lahmenCSV.to_csv("relevant_stats.csv", index=False)

#relevant_Stats_csv = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/relevant_stats.csv')

#get_stats_for_year(1888)
#print("average stats")
#calculate_average(get_stats_for_year(1888))
#add_average_stats_to_csv(calculate_average(get_stats_for_year(1888)))

#add_average_stats_for_range_of_years_to_csv( 1888, 2015)
#recall_average_stats_for_a_given_year(2000)
#recall_team_stats_for_a_given_year(2000)
#compare_teamStats_to_leauge_Ave(2000)
#add_teamStats_to_leauge_ave_for_range (1888, 2015)

#write_teamStats_to_leauge_ave_to_csv(add_classification_column_to_data())

#transformations added additional unneeded columns to the data this gets rid of them
write__excess_columns_removed_teamStats_to_leauge_ave_to_csv(remove_excess_columns_classification_column_to_data())







