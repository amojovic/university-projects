import random
import math
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt

DEFAULT_NUM_RESTARTS = 1
DEFAULT_POPULATION_SIZE = 100
DEFAULT_NUM_MATING_INDIVIDUALS = 50
DEFAULT_NUM_MUTATING_INDIVIDUALS = 30
DEFAULT_NUM_INDIVIDUALS_TO_ELIMINATE = 30
DEFAULT_RANDOM_SEED = 77
DEFAULT_OUTPUT_FILE = "rezultati.txt"
DEFAULT_X_BOUNDS = [-1.5, 4]
DEFAULT_Y_BOUNDS = [-3, 4]
DEFAULT_MUTATION_RATE = 0.2
DEFAULT_BIT_LENGTH = 16
DEFAULT_GENERATIONS = 1000


def load_config(config_file=None):         # omogucavam da ne bude prosledjen config file
    config = {}                            # dictcionary za parametre koje cu koristiti

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        print("Nije dat custom config, koristimo default vrednosti.")

    defaults = {
        "population_size": DEFAULT_POPULATION_SIZE,
        "num_mating_individuals": DEFAULT_NUM_MATING_INDIVIDUALS,
        "num_mutating_individuals": DEFAULT_NUM_MUTATING_INDIVIDUALS,
        "num_individuals_to_eliminate": DEFAULT_NUM_INDIVIDUALS_TO_ELIMINATE,
        "num_restarts": DEFAULT_NUM_RESTARTS,
        "random_seed": DEFAULT_RANDOM_SEED,
        "output_file": DEFAULT_OUTPUT_FILE,
        "x_bounds": DEFAULT_X_BOUNDS,
        "y_bounds": DEFAULT_Y_BOUNDS,
        "mutation_rate": DEFAULT_MUTATION_RATE,
        "bit_length": DEFAULT_BIT_LENGTH,
        "generations": DEFAULT_GENERATIONS
    }

    # ovde sada kupim parametre koji su relevantni za proveru greski, npr. ne moze da se pari vise jedinki od broja cele populacije
    population_size = config.get("population_size", DEFAULT_POPULATION_SIZE)
    num_mating_individuals = config.get("num_mating_individuals", DEFAULT_NUM_MATING_INDIVIDUALS)
    num_mutating_individuals = config.get("num_mutating_individuals", DEFAULT_NUM_MUTATING_INDIVIDUALS)
    num_individuals_to_eliminate = config.get("num_individuals_to_eliminate", DEFAULT_NUM_INDIVIDUALS_TO_ELIMINATE)
    mutation_rate = config.get("mutation_rate", DEFAULT_MUTATION_RATE)

    if (num_mating_individuals > population_size or 
        num_mutating_individuals > population_size or 
        num_individuals_to_eliminate > population_size or
        num_individuals_to_eliminate > num_mating_individuals):
        
        print("Detektovane greske u konfiguraciji. Koristimo sve default vrednosti.")
        return defaults

    if not (0 <= mutation_rate <= 1):
        print("Greska u vrednosti mutation_rate. Mora biti između 0 i 1. Koristimo default vrednost.")
        mutation_rate = DEFAULT_MUTATION_RATE

    # setujem ceo dictionary, ako fali vrednost stavim default
    return {
        "num_restarts": config.get("num_restarts", defaults["num_restarts"]),
        "population_size": population_size,
        "num_mating_individuals": num_mating_individuals,
        "num_mutating_individuals": num_mutating_individuals,
        "num_individuals_to_eliminate": num_individuals_to_eliminate,
        "random_seed": config.get("random_seed", defaults["random_seed"]),
        "output_file": config.get("output_file", defaults["output_file"]),
        "x_bounds": config.get("x_bounds", defaults["x_bounds"]),
        "y_bounds": config.get("y_bounds", defaults["y_bounds"]),
        "mutation_rate": mutation_rate,
        "bit_length": config.get("bit_length", defaults["bit_length"]),
        "generations": config.get("generations", defaults["generations"])
    }

# dekodovanje koraci:  
# binarni [1, 0, 1, 1] mapira u string ['1', '0', '1', '1']
# string ['1', '0', '1', '1'] join-uje '1011'
# cita '1011' kao int sto znaci 11 decimalno
# skaliram je a dat interval, scale_factor = (upper_bound - lower_bound) / max_value
# max_value = 2**len(chromosome) - 1, jer binarno 2^n - 1 je max, 1111 na 4 bita je 15 dok 2^n bi bilo 16
# realna_vrednost = lower_bound + decimal_value * scale_factor

def decode_chromosome(chromosome, lower_bound, upper_bound):
    decimal_value = int(''.join(map(str, chromosome)), 2)
    return lower_bound + decimal_value * (upper_bound - lower_bound) / (2**len(chromosome) - 1)

# Mekormikova funkcija

def fitness_function(x, y):
    return math.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1

# evaluacija = dekodovanje x i y hromozoma + primena funkcije da vrati prilagodjenost

def evaluate_population(population, x_bounds, y_bounds):
    fitness_values = []
    for chrom in population:
        x_chrom = chrom[:DEFAULT_BIT_LENGTH]
        y_chrom = chrom[DEFAULT_BIT_LENGTH:]
        
        x = decode_chromosome(x_chrom, x_bounds[0], x_bounds[1])
        y = decode_chromosome(y_chrom, y_bounds[0], y_bounds[1])
        
        fitness_values.append(fitness_function(x, y))
    return fitness_values

# selekcija po redosledu sa vezbi, jedinka 2i-1 su uprauje sa 2i ali uparivanje radim u glavnoj funckiji
# argsort vraca redosled indeksa kojim treba da se sortira
# ako su fitnessi [7.67, -1.93, 2.45] i populacija npr. [0001, 0011, 0100]
# onda argsort fitnessa vraca [1, 2, 0] jer -1.93 na indeksu 1 je najmanji itd.
# for loopom onda sortiram populaciju po indeksima [0011, 0100, 0001]

def selection(population, fitness_values, num_mating_individuals):

    sorted_indices = np.argsort(fitness_values)
    sorted_population = []

    for i in sorted_indices:
        sorted_population.append(population[i])


    selected_population = sorted_population[:num_mating_individuals]  # hvatamo onoliko njih koliko config kaze da treba da se pare

    if len(selected_population) % 2 != 0:   # ako neparan broj, izbacimo jedu jedinku
        selected_population.pop()

    return selected_population

# uniformno ukrstanje, za svaku poziciju se nasumicno bira od kogog roditelja prima gen, 50%

def crossover(parent1, parent2):
    child1 = []
    child2 = []
    
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])

    return child1, child2

# tackasta mutacija flipovanje random bitova, sto je veci mutation_rate vece su sanse da se obrne taj bit

def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


# generise populaciju (lista) random binarnih hromozoma, [ [0,0,1,0], [1,0,1,0], [0,1,1,0] ] // primer populacije 3 jedinki x hromozom duzine 2, y duzine 2

def initialize_population(population_size, bit_length):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, 1) for _ in range(bit_length * 2)]  
        population.append(chromosome)
    return population



def genetic_algorithm(config):
    random.seed(config['random_seed'])              # omogucavam random vrednost na osnovu configa

    all_best_fitness = []             # Za grafik za sve restarte
    all_average_fitness = []

    with open(config['output_file'], 'w') as f:     # zapocinjem pisanje rezultata u output file

        for restart in range(config['num_restarts']): 
            f.write(f"Restart broj {restart + 1}/{config['num_restarts']}\n")

            population = initialize_population(config['population_size'], config['bit_length'])

            best_fitness_history = []             # Za grafik za sve generacije
            average_fitness_history = []

            for generation in range(config['generations']): 
                print(f"Generation {generation}: Population size = {len(population)}")
                fitness_values = evaluate_population(population, config['x_bounds'], config['y_bounds'])

                best_fitness = min(fitness_values)  
                average_fitness = np.mean(fitness_values)

                best_fitness_history.append(best_fitness)            # Cuvam ovde
                average_fitness_history.append(average_fitness)


                f.write(f"Generacija {generation}: Najbolja prilagodjenost: {best_fitness}, Prosecna prilagodjenost: {average_fitness}\n")

                selected_population = selection(population, fitness_values, config['num_mating_individuals'])
              
                sorted_indices = np.argsort(fitness_values) 
                elite_population = []                              # sortiram od najboljih do najgorih
                for i in sorted_indices:
                    elite_population.append(population[i])

                elite_population = elite_population[:-config['num_individuals_to_eliminate']]   # izbacujem onoliko najgorih elemenata koliko pise u configu
                new_population = elite_population.copy()  

                children_start_index = len(new_population)

                # ovde je primenjen princip selekcije po redosledu gde se uparuju 1 i 2, 3 i 4 itd.
                i = 0
                while len(new_population) < config['population_size']:
                    parent1 = selected_population[i] 
                    parent2 = selected_population[i + 1]
                    child1, child2 = crossover(parent1, parent2)
                    new_population.append(child1)
                    if len(new_population) < config['population_size']:
                        new_population.append(child2)
                    i = i+2

                # samo mutiram decu, jer ako mutiram roditelje postoji mogucnost da ostetim elitnu jedinku i pokvarim rezultat (bioloski gledano oba takodje ima vise smisla)

                num_children = len(new_population) - children_start_index
                num_to_mutate = min(config['num_mutating_individuals'], num_children)

                for i in random.sample(range(children_start_index, len(new_population)), num_to_mutate):
                    mutate(new_population[i], config['mutation_rate']) 

                new_fitness_values = evaluate_population(new_population, config['x_bounds'], config['y_bounds'])
                sorted_indices = np.argsort(new_fitness_values)  
                new_population = [new_population[i] for i in sorted_indices]  

                population = new_population


            best_index = np.argmin(new_fitness_values)  
            best_chromosome = population[best_index]
            best_x = decode_chromosome(best_chromosome[:config['bit_length']], config['x_bounds'][0], config['x_bounds'][1])
            best_y = decode_chromosome(best_chromosome[config['bit_length']:], config['y_bounds'][0], config['y_bounds'][1])

            f.write(f"Najbolji hromozom: {best_chromosome}\n")
            f.write(f"Fenotipske vrednosti: x = {best_x}, y = {best_y}\n")
            f.write(f"Funkcija: {fitness_function(best_x, best_y)}\n")

            all_best_fitness.append(best_fitness_history)              # Dodajem ovde cele liste
            all_average_fitness.append(average_fitness_history)


    plt.figure(figsize=(12, 6))        # sirina 12 , visina 6 cele figure

    plt.subplot(1, 2, 1)               # 1 red, 2 kolone, prvi subplot
    for i, best_fitness in enumerate(all_best_fitness):
        plt.plot(best_fitness, label=f'Restart {i + 1}')
    plt.xlabel('Generacija')
    plt.ylabel('Najbolja prilagodjenost')
    plt.title('Najbolja prilagodjenost po generaciji')
    plt.legend()

    plt.subplot(1, 2, 2)               # 1 red, 2 kolone, drugi subplotho
    for i, average_fitness in enumerate(all_average_fitness):
        plt.plot(average_fitness, label=f'Restart {i + 1}')
    plt.xlabel('Generacija')
    plt.ylabel('Prosecna prilagodjenost')
    plt.title('Prosecna prilagodjenost po generaciji')
    plt.legend()
    plt.savefig('grafik.png')
    plt.close()        

parser = argparse.ArgumentParser()
parser.add_argument('config_file', nargs='?', type=str)   # namestam parser da bi mogao da primi custom config file, nargs znaci da je opcionalan
args = parser.parse_args()
    
config_file = args.config_file if args.config_file else None    # ako ima custom config file onda njega ako ne nista pa ce biti default
config = load_config(config_file)
    
genetic_algorithm(config)
