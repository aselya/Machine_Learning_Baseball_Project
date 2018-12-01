'''
chromosome (collection of genes)
    list of genes
    set gene value (index, subject)

'''
import random
import pandas as pd
import numpy as np

POPULATION_SIZE = 10
NUMBER_OF_FITTEST_CHROMOSOMES = 2
TOURNAMENT_SIZE = 4
RATE_OF_MUTATION = 0.05
GOAL_TIME = 120
GENERATIONS = 10
BEST_POSSIBLE_FITTNESS_SCORE = 1000

SUBJECTS = ['math', 'science', 'english']
question_time = pd.Series([])
question_database = pd.DataFrame( columns=[ 'subject', 'time', 'used'], )




for i in range(10):
    question_database['time'] = [np.random.randint(5,10) for n in range(10)]
    question_database['subject'] = np.random.randint(0,3, question_database.shape[0])
    question_database['used'] = False




#print(question_database)

question_database_array = [['math', 10, False],['math', 9, False],['math', 7, False],['math', 3, False],['math', 10, False],['math', 9, False],['math', 7, False],['math', 3, False],
                           ['science', 10, False],['science', 9, False],['science', 7, False],['science', 3, False], ['science', 10, False],['science', 9, False],['science', 7, False],['science', 3, False],
                           ['english', 10, False],['english', 9, False],['english', 7, False],['english', 3, False], ['english', 10, False],['english', 9, False],['english', 7, False],['english', 3, False],
                           ]
#print(question_database_array) #checks database array
#print(question_database_array[2][1]) # checks to make sure the database array can be accessed as expected
#print(question_database_array[2])


class Chromosome:
        def set_fitness_score(self):

            #fitness function is squared difference between the individual scores of the subjects and average
            #plus the squared difference between goal time and actual time
            #all subjtected from 10000
            fittness_score = BEST_POSSIBLE_FITTNESS_SCORE
            goal_time_per_subject = GOAL_TIME/len(SUBJECTS)
            #print("goal time per subject: " + str(goal_time_per_subject))
            fittness_score = fittness_score - (self.english_time - goal_time_per_subject)**2
            #print("fittness_score english: " + str(fittness_score))
            fittness_score = fittness_score - (self.math_time - goal_time_per_subject)**2
            #print("fittness_score math: " + str(fittness_score))
            fittness_score = fittness_score - (self.science_time - goal_time_per_subject)**2
            #print("fittness_science: " + str(fittness_score))
            return fittness_score


        def get_gene_chromosomes(self):
            return self.gene_chromosomes

        def get_new_question(self, database ):
            random_index = question_database_array[random.randint(0, len(question_database_array)-1)]
            return random_index
        def __init__(self):
            self.gene_chromosomes = []
            self.fittness_score = 0
            self.total_time_for_chromosome = 0
            self.math_time = 0
            self.science_time = 0
            self.english_time = 0

            while self.total_time_for_chromosome < GOAL_TIME:
                new_question = self.get_new_question(question_database_array)
                #print("total time" + str(self.total_time_for_chromosome))
                self.total_time_for_chromosome = new_question[1] + self.total_time_for_chromosome
                #print('new time' + str(self.total_time_for_chromosome))
                self.gene_chromosomes.append(new_question)
                if new_question[0] == 'math':
                    #print("math question selected")
                    self.math_time = new_question[1] + self.math_time
                if new_question[0] == 'science':
                    #print("science question selected")
                    self.science_time = new_question[1] + self.science_time
                if new_question[0] == 'english':
                    #print("science question selected")
                    self.english_time = new_question[1] + self.english_time
            #print("english time" + str(self.english_time))
            #print("science time" + str(self.science_time))
            #print("math time" + str(self.math_time))
            '''
            if self.total_time_for_chromosome != (self.math_time + self.science_time +self.english_time):
               print("total time does not match subject times error has occured")
            else:
              print("total and subject times match")
            '''
            self.fittness_score = self.set_fitness_score()
            #print("chrom fittness score = " + str(self.fittness_score))


           # print(self.gene_chromosomes)
           # print(len(self.gene_chromosomes))
#end chromosome
chrom1 = Chromosome()

class Population:
    #population contains population of several chromesomes

    population_of_chromosomes = [] #where they will be housed
    top_fittness_score = 0

    current_gen = 0

    def cross_over(chrom1 , chrom2):
        new_chromosome = Chromosome()
        #print(new_chromosome.gene_chromosomes[1])
        count = 0
        for question in new_chromosome.gene_chromosomes:
            random_value = random.randint(0,10)
            #print("random value" + str(random_value))
            if random_value > 5:
                #print(str(new_chromosome.gene_chromosomes[count]) +'is now' + str(chrom1.gene_chromosomes[count]))
                new_chromosome.gene_chromosomes[count] = chrom1.gene_chromosomes[count]
                #print(str(chrom1.gene_chromosomes[count]))
            else:
                new_chromosome.gene_chromosomes[count] = chrom2.gene_chromosomes[count]
        new_chromosome.set_fitness_score()
        #print(str(new_chromosome.set_fitness_score()))
        return new_chromosome

    def set_up_mutation(population_of_chromosomes):
        count = 0
        while count < len(population_of_chromosomes):
            current = population_of_chromosomes[count]
            #print("current: " + str(current))
           # print("current" + str(current.get_gene_chromosomes()))
            current_chrom = current.get_gene_chromosomes()
         #mutation_gene = current_chrom[count]
        # print("current" + str(current.get_gene_chromosomes()))
           # print("####### current chrom value: " + str(current_chrom) + "###################")
        #mutation_of_chromosomes(current_chrom)

            for i in current_chrom:
                chrom = Chromosome
                if random.random() < RATE_OF_MUTATION:
                    #print("before mutation" +str(current_chrom[count]))
                    current_chrom[count] = chrom.get_new_question(chrom, question_database_array)
                   # print("after mutation" +str(current_chrom[count]))
            count += 1

    while len(population_of_chromosomes) < POPULATION_SIZE:
                population_of_chromosomes.append(Chromosome())

    while current_gen < GENERATIONS:
        if top_fittness_score < BEST_POSSIBLE_FITTNESS_SCORE:
            current_gen += 1

            #print("sorting the scores")
            population_of_chromosomes.sort(key=lambda x: x.fittness_score, reverse=True)
            #for chrom in population_of_chromosomes:
                #print(str(chrom.fittness_score))

            top_fittness_score = population_of_chromosomes[0].fittness_score
            #print("top fitness score" + str(top_fittness_score))
            population_of_chromosomes = population_of_chromosomes[:2]

            while len(population_of_chromosomes) < POPULATION_SIZE:
                population_of_chromosomes.append(Chromosome())
                #print(str(population_of_chromosomes))

            #for chrom in population_of_chromosomes:
                #print(str(chrom.fittness_score))



    #get parents
            #population_of_chromosomes = population_of_chromosomes[0:2]
                print(str(population_of_chromosomes))

                population_of_chromosomes.append(cross_over(population_of_chromosomes[0], population_of_chromosomes[1]))

            #print(str(population_of_chromosomes))
                set_up_mutation(population_of_chromosomes)
            print("Generation: " + str(current_gen) + " top fittness score: " + str(top_fittness_score))
            print("pop length" + str(len(population_of_chromosomes)))
