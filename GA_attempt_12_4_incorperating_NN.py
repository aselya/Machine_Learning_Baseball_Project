import pandas as pd
import random
import premade_estimator
from premade_estimator import classifier


'''

fitness equation
playoff team = 200
good team = 150
average team = 100
bad team = 50

since the goal is to find the stats that make the most difference while still making the playoffs
points will be rewarded for any stat below average since that indicates the stat has no significant impact upon wins
and shouldn't be what teams focus on
conversely any stat below average will be rewarded with a higher contribution fitness score
The base rates are far apart enough that there shouldn't be a chance for a bad team to recieve a high enough score to be a playoff team
ideal is 1
for each stat if the stat is > 1
    take difference squared between stat and ideal and subtract from base score
for each stat <= 1
    take difference squared from stat and ideal and add it to the base score


starting breeding population will be 20 randomly chosen from existing data
top 4 will be parents

'''
GENERATIONS = 10
RANDOM_SAMPLE_SIZE = 20
DATASET = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/lahmenCSVmanipulation/venv/team_stats_ave_no_excess_stats.csv', index_col=None)
POPULATION_TEAMS = []
IDEAL_POPULATION_SIZE = 20

def get_random_sample():
    random_subset = DATASET.sample(n=RANDOM_SAMPLE_SIZE)
    #drop lable column, that will be determined by NN
    random_subset1 = random_subset.drop(columns=['Classification'])
    print("columns" + str(len(random_subset1.columns)))
    NUMBER_OF_STATS_TRACKED = len(random_subset1.columns)
    print("number stat tracked: " + str(NUMBER_OF_STATS_TRACKED))
    print(random_subset1.head(10))
    return random_subset1

BASE_POPULATION = get_random_sample()
print("base population length: " + str(len(BASE_POPULATION)))

NUMBER_OF_STATS_TRACKED = len(BASE_POPULATION.columns)
print("number of stats tracked" + str(NUMBER_OF_STATS_TRACKED))


class Team_chromosomes:

    def make_new_stat_value(self):
        new_value = 1 + random.randint(-5000,5000)/100000
        print("new stat value" + str(new_value))
        return new_value

    def add_neural_net_classification(self):
        '''
        call neural net on data return classification
        if classification = playoff return 200
        if classification = good return 150
        if classification = ave return 100
        if classification = bad return 50
        '''
        neural_net

    def format_chromosome_as_dict(self, chromosome):
        print("format_chromosome_as_dict")
        default_value = 1
        keys = ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP']
        dictionary = dict.fromkeys(['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP'],default_value)
        count = 0
        for gene in chromosome:
            dictionary[keys[count]] = [gene]
            count += 1
        print("dictionary" + str(dictionary))

        return dictionary

    def get_nn_classification(self, dict):
        pred = premade_estimator.get_prediction(classifier, dict)
        #pred = neural_net.classifier.predict(input_fn=lambda:iris_data.eval_input_fn(predict_x3,labels=None,batch_size=args.batch_size))
        print(pred)
        print("prediction:" + str(pred))
        return pred


    def set_fitness_score(self, chromosome, NN_prediction):
        print("set_fittness_score" + str(NN_prediction))
        prediction_string = str(NN_prediction)
        score = 0
        count = 0
        if prediction_string == "Bad Team":
            score = score + 50
        if prediction_string == "Average Team":
            score = score +100
        if prediction_string == "Good Team":
            score = score +150
        if prediction_string == "Playoff Team":
            score = score +200

        print("set score:")

        for gene in chromosome:
            count += 1
            if gene > 1:
                score = score - gene**2
            else:
                score = score + gene**2

        #score = score/count
        #score = score + self.add_neural_net_classification()
        print("score is : " + str(score))

        return score
    '''
    #one point crossover in the middle
    def cross_over(self, parent1, parent2):
        new_team_chromosome = Team_chromosomes() #makes new chromosome
        count = 0
        for gene in new_team_chromosome.genes_made_of_stats:
            if count > NUMBER_OF_STATS_TRACKED/2:
                new_team_chromosome.genes_made_of_stats[count] = parent1.genes_made_of_stats[count]
            else:
                new_team_chromosome.genes_made_of_stats[count] = parent2.genes_made_of_stats[count]
        new_team_chromosome.set_fitness_score()
        format_chromosome_as_dict(new_team_chromosome)
        return new_team_chromosome
    '''

    def __init__(self, length):
        self.genes_made_of_stats = []
        self.fittness_score = 0
        self.current_length = 0
        self.fittness_score = 0
        print("new chromosome created")
        print("length" + str(length))
        print("current length :" + str(self.current_length) + " numb stats tracked" + str(length))
        while self.current_length < length:
            new_gene = self.make_new_stat_value()
            self.genes_made_of_stats.append(new_gene)
            self.current_length += 1
        prediction = self.get_nn_classification(self.format_chromosome_as_dict(self.genes_made_of_stats))

        self.fittness_score = self.set_fitness_score(self.genes_made_of_stats, prediction)


#end chromosome class
#one point crossover in the middle
def format_chromosome_as_dict(chromosome):
        print("format_chromosome_as_dict")
        default_value = 1
        keys = ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP']
        dictionary = dict.fromkeys(['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP'],default_value)
        count = 0
        for gene in chromosome.genes_made_of_stats:
            dictionary[keys[count]] = [gene]
            count += 1
        print("dictionary" + str(dictionary))

        return dictionary

def cross_over( parent1, parent2):
    new_team_chromosome = Team_chromosomes(NUMBER_OF_STATS_TRACKED) #makes new chromosome
    count = 0
    for gene in new_team_chromosome.genes_made_of_stats:
        if count > NUMBER_OF_STATS_TRACKED/2:
            new_team_chromosome.genes_made_of_stats[count] = parent1.genes_made_of_stats[count]
        else:
            new_team_chromosome.genes_made_of_stats[count] = parent2.genes_made_of_stats[count]
    prediction = new_team_chromosome.get_nn_classification(new_team_chromosome.format_chromosome_as_dict(new_team_chromosome.genes_made_of_stats))


    new_team_chromosome.set_fitness_score(new_team_chromosome.genes_made_of_stats, prediction)
    format_chromosome_as_dict(new_team_chromosome)
    return new_team_chromosome

def build_POPULATION_TEAMS( population):
    NEW_POPULATION_TEAMS = []
    while len(NEW_POPULATION_TEAMS) < IDEAL_POPULATION_SIZE:
        print("numbrer of stats tracked " +str(NUMBER_OF_STATS_TRACKED))
        NEW_POPULATION_TEAMS.append(Team_chromosomes(NUMBER_OF_STATS_TRACKED))
    NEW_POPULATION_TEAMS.sort(key=lambda x: x.fittness_score, reverse=True)
    for chrom in POPULATION_TEAMS:
        print(str(chrom.fittness_score))

        NEW_POPULATION_TEAM = NEW_POPULATION_TEAMS[0:4]
        randint = random.randint(0,4)
        randint2 = random.randint(0,4)
        while randint == randint2:
            randint = random.randint(0,4)

        NEW_POPULATION_TEAMS.append(cross_over(NEW_POPULATION_TEAMS[randint], NEW_POPULATION_TEAMS[randint2]))
        #population_of_chromosomes.append(cross_over(population_of_chromosomes[0], population_of_chromosomes[1]))

    return NEW_POPULATION_TEAMS
count = 0
while count < GENERATIONS:
    POPULATION_TEAMS = build_POPULATION_TEAMS(POPULATION_TEAMS)
    count += 1
    print("generations:" + str(count))
#makes the first 4 the parents
    POPULATION_TEAMS.sort(key=lambda x: x.fittness_score, reverse=True)
    print(str(POPULATION_TEAMS))
    for chrom in POPULATION_TEAMS:
        print(str(chrom.fittness_score))





