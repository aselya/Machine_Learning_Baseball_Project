'''
This algorithim is my first implimentation of a genetic algorithim using a binary chromosome
In the next iteration of the algorithim the goal chromosome will be replaced with a neural network
The neural network will be trained on baseball data and the chromosome structure will be the inputs passed to the neural net
with each 1 representing a value of greater than leauge average on a given year

'''


import random
from random import randint

POPULATION_SIZE = 10
GOAL_CHROMOSOME = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
NUMBER_OF_FITTEST_CHROMOSOMES = 2
TOURNAMENT_SIZE = 4
RATE_OF_MUTATION = 0.05

'''
chromosomes are the structures that are used to store genes in the GA
When they are constructed they generate binary randomly
'''
class Chromosome:
    def __init__(self):
        self._genes = []
        self._fitness = 0
        i = 0
    #randomly populates array for chromosones based on length of the target
        while i < GOAL_CHROMOSOME.__len__():
            if random.random() >= .5:
                self._genes.append(1)
            else:
                self._genes.append(0)
            i += 1

    def get_genes(self):
        return self._genes

    #determines fitness and returns value
    def get_fitness(self):
        self._fitness = 0
        #for loop compares genes in chromosome to ideal
        for i in range(self._genes.__len__()):
            if self._genes[i] == GOAL_CHROMOSOME[i]:
                    self._fitness += 1
        return self._fitness

    #to string method
    def __str__(self):
            return self._genes.__str__()

    #population is the collection of genes used
class Population:
    #takes in size of population and creates enough chromosomes to fit population
    def __init__(self, size):
        self._chromosomes =[ ]
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome())
            i += 1

    #returns chromosomes in current population
    def get_chromosomes(self):
        return self._chromosomes


class GeneticAlgorithim:

    @staticmethod
    #mutates the population of the GA based on the results of the return of the _crossover_population method
    def evolve(pop):
        return GeneticAlgorithim._mutate_population(GeneticAlgorithim._crossover_population(pop))

    '''
    isolates the fittest chromosome and ensures they continue in population
    creates new chromosomes by filling remaining population spots with children consiting of traits from fittest
    '''
    @staticmethod
    def _crossover_population(pop):
        crossover_pop = Population(0)
        for i in range(NUMBER_OF_FITTEST_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = NUMBER_OF_FITTEST_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithim._select_tournament_population(pop).get_chromosomes()[0] #gets fittest from prev generation
            '''
            conditional statment checks if there parameters call for retention of more than one fit chromosome
            if so it randomly selects one of the fit ones as a parent otherwise it uses the initial one
            '''
            if NUMBER_OF_FITTEST_CHROMOSOMES > 1 and GeneticAlgorithim._select_tournament_population(pop).get_chromosomes()[NUMBER_OF_FITTEST_CHROMOSOMES] is not None:
                chromosome2 = GeneticAlgorithim._select_tournament_population(pop).get_chromosomes()[randint(0, NUMBER_OF_FITTEST_CHROMOSOMES)] #gets next fittest
            else:
                chromosome2 = GeneticAlgorithim._select_tournament_population(pop).get_chromosomes()[0] #gets next fittest

            crossover_pop.get_chromosomes().append(GeneticAlgorithim._crossover_chromosomes(chromosome1, chromosome2))
            i += 1

        return crossover_pop

    @staticmethod
    def _mutate_population(pop):
        for i in range(NUMBER_OF_FITTEST_CHROMOSOMES, POPULATION_SIZE):
        #range makes sure that fittest chromosomes aren't impacted by mutation only their offspring
            GeneticAlgorithim._mutate_chromosome(pop.get_chromosomes()[i])

        return pop

    @staticmethod
    #uses random to determine what chromosome will contribute a gene to a new population
    def _crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Chromosome()
        for i in range(GOAL_CHROMOSOME.__len__()):
            if random.random() >= 0.5:
                crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
            else:
                crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]

        return crossover_chrom

    @staticmethod
    #uses random to determine if mutation of one of the genes occurs
    def _mutate_chromosome(chromosome):
        for i in range(GOAL_CHROMOSOME.__len__()):
            if random.random() < RATE_OF_MUTATION:
                if random.random() < 0.5:
                  chromosome.get_genes()[i] = 1
                else:
                    chromosome.get_genes()[i]= 0


    '''
    impliments a tournament selection scheme
    where the chromosomes are randomlly slected from the population
    compared by fittest and returned for use in crossover
    '''
    @staticmethod
    def _select_tournament_population(pop):
        tournament_pop = Population(0)
        i = 0
        while i < TOURNAMENT_SIZE:
                tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
                i += 1
        tournament_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_pop



#print method
def _print_population(pop, gen_number):
    print("\n----------------------------")
    print("Generation #", gen_number, " | Fittest chromosome fittness:", pop.get_chromosomes()[0].get_fitness())

    print("Targe chromosome:", GOAL_CHROMOSOME)
    print("-------------------------------")

    i = 0
    for x in pop.get_chromosomes():
        print("Chromosomes #", i, " :", x, "| Fitness ", x.get_fitness())
        i += 1


population = Population(POPULATION_SIZE)
population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse= True)
_print_population(population, 0)
generation_number = 1
while population.get_chromosomes()[0].get_fitness() < GOAL_CHROMOSOME.__len__():
    population = GeneticAlgorithim.evolve(population)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse= True)
    _print_population(population, generation_number)
    generation_number +=1
