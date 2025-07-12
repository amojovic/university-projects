import random
import sys
from collections import defaultdict
import math
import time

class Graph:
    def __init__(self, n):
        self.n = n
        self.edges = defaultdict(dict)

#   self.edges = {
#    0: {1: 1, 2: 5, 3: 7},
#    1: {0: 1, 2: 2, 3: 3},
#    2: {0: 5, 1: 2, 3: 9},
#   itd....
#   }
#
#   ovako izgleda, znaci recnik recnika
#   prvi broj je cvor, i onda unutra su tezine ivice do drugog cvora
    
    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight
    # ovo mi je da dodam tezine za odredjene ivice, neusmeren je graf tako da mora u oba smera
    # poziv npr. g.add_edge(0, 1, 2.0) ce da dodata 0:{1: 2.0} i 1:{0: 2.0}
    
    def get_weight(self, u, v):
        return self.edges[u].get(v, float('inf')) # vraca tezinu izmedju dva cvora ako ima, ako ne ond infinite

def read_graph(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n = int(lines[0].strip())    # onaj broj u prvoj liniji, broj cvorova
    g = Graph(n)                 # sada ide instanca grafa
    
    for i in range(n-1):    # sada ostale linije
        weights = list(map(float, lines[i+1].strip().split())) # svi se citaju kao floatovi i pravim niz, i+1 jer je prva bila broj cvorova
        for j in range(len(weights)):  # sada povezujem taj cvor sa ostalim cvorovima na osnovu te linije
            u = i
            v = i + 1 + j
            g.add_edge(u, v, weights[j])
    
    return g

class Individual:
    def __init__(self, edges, graph):
        self.edges = edges    # grane, tj. parovi u v
        self.graph = graph    # referenca na sam graf
        self.fitness = None   # ovo će mi biti suma težina svih grana
    
    def calculate_fitness(self):
        if self.fitness is None:
            self.fitness = sum(self.graph.get_weight(u, v) for u, v in self.edges) # težine svih grana iz rešenja se računa i sabira
        return self.fitness
    
    def is_valid_tree(self):
        # MST po definiciji ima n-1 grana tako da radim proveru
        if len(self.edges) != self.graph.n - 1:
            return False
        
        # moram da proverim povezanost, koristiću BFS
        adj = defaultdict(list)  # pravim sličnu strukturu kao gore za graf
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # strukture za graf, nijedan čvor na početku nije pipnut
        visited = [False] * self.graph.n
        queue = [0]
        visited[0] = True
        count = 1
        
        # klasičan BFS
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == self.graph.n  # ovo će biti True ako jeste povezan

def initialize_population(graph, population_size):
    population = []
    
    for _ in range(population_size):
        edges = set()  # prazan set ivica, grana štagod. Neće biti duplikata
        nodes = set([random.randint(0, graph.n-1)])  # random početni čvor
            
        while len(edges) < graph.n - 1:
            u = random.choice(list(nodes))
            possible_v = [(v, graph.get_weight(u, v)) for v in range(graph.n) if v not in nodes]
            # pokušavam da nađem drugi čvor koji još nije u drvetu, ide redom i uzima težine 
            if not possible_v:
                break
            # ako nema onda smo završili
              
            # da bi dobili neki solidan rezultat, dajem preferencu manjim težinama
            # znači sort po težinama (manje su na početku)  
            possible_v.sort(key=lambda x: x[1])

            # random daje neki broj između 0 i 1, tako da to puta dužina i onda koren kako bi generalno težili početku niza, min je samo da ne iskoči van opsega
            idx = min(int(math.sqrt(random.random() * len(possible_v))), len(possible_v)-1)

            # onda taj dodamo
            v = possible_v[idx][0]
            edges.add((min(u, v), max(u, v)))
            nodes.add(v)
        
        population.append(Individual(edges, graph))
    
    return population

# pravimo uniju ivica (bukvalno sve ivice od oba roditelja bez duplikata)
# sortira ih po weightovima rastuće
# koristim Union-Find (Disjoint Set Union - DSU), moram jer je bukvalno presporo na bilo koji drugi način koji sam probao
# to je ova parent lista gde je svako sebi parent na početku
# poenta je da se proveri da li pravi ciklus

def crossover(parent1, parent2):
    edge_union = parent1.edges.union(parent2.edges)
    edge_list = sorted(edge_union, key=lambda e: parent1.graph.get_weight(e[0], e[1]))
    
    parent = [i for i in range(parent1.graph.n)]
    
    # ako nije sam sebi parent ide dublje gore, pravi shortcutove usput
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    new_edges = set()
    # proverava da li čvorovi pripadaju istoj grupi, 
    # ako različiti to znači da spaja dve različite grupe i ne pravi ciklus tako da je to novi skup ivica
    for edge in edge_list:
        u, v = edge
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            new_edges.add(edge)
            parent[root_v] = root_u
            if len(new_edges) == parent1.graph.n - 1:
                break
    
    return Individual(new_edges, parent1.graph)

def mutate(individual, mutation_rate):
    if random.random() > mutation_rate:
        return individual
    
    # ivice u listu za lakši rad
    # provera da ne mutiramo ako ima samo jedna lol
    edges = list(individual.edges)
    if len(edges) < 2:
        return individual
    
    # ovim praktično cepamo stablo na dva dela jer izbacimo neku ivicu, realno u MST su sve bitne
    edge_to_remove = random.choice(edges)
    edges.remove(edge_to_remove)
    
    # opet moram susednost
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
   
    # za prvu komponentu tražimp pomoću DFS gde se desio presek to jest poslednji čvor od one ivice
    component1 = set()
    stack = [edge_to_remove[0]]
    while stack:
        u = stack.pop()
        if u not in component1:
            component1.add(u)
            for v in adj[u]:
                stack.append(v)
    
    # ovo su ostali
    component2 = set(range(individual.graph.n)) - component1
   
    # lista kandidata da bi opet spojili, znači sve moguće kombinacije da se poveže nešto is comp1 i comp2
    candidate_edges = []
    for u in component1:
        for v in component2:
            candidate_edges.append((min(u, v), max(u, v), individual.graph.get_weight(u, v)))
    
    # ako ne može ništa onda vraćamo staru
    if not candidate_edges:
        return individual
    
    # soritramo po težini i uzmemo najbolju
    candidate_edges.sort(key=lambda x: x[2])
    new_edge = candidate_edges[0][:2]
    edges.append(new_edge)
    
    return Individual(set(edges), individual.graph)

# klasična turnirnska selekcija, biramo učesnike nasumično
# od ovih uzimam onaj sa najmanjom težinom 

def tournament_selection(population, tournament_size):
    selected = random.sample(population, tournament_size)
    return min(selected, key=lambda x: x.calculate_fitness())

# glavni deo, u suštini koristi sve ostale funkcije redom
# generiše populaciju jedinki na osnovu grafa
# bira najbolju jedinku
# iterira kroz generacije
# turnirska selekcija + crossover + mutacija dece
# is valid tree provera za decu
# provera na kraju da li je trenutno rešenje bolje od prethodnog

def evolutionary_algorithm(graph, population_size=100, generations=200, 
                          mutation_rate=0.2, tournament_size=5):

    population = initialize_population(graph, population_size)

    best_individual = min(population, key=lambda x: x.calculate_fitness())
    # mala lambdica, poziva calc fit iz Indiviudal klase
    
    for generation in range(generations):
        new_population = []
        
        # elitizam malo ovde, čuvam najboljeg za sledeću generaciju
        new_population.append(best_individual)
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            
            if child.is_valid_tree():
                new_population.append(child)
        
        population = new_population
        current_best = min(population, key=lambda x: x.calculate_fitness())
        
        if current_best.calculate_fitness() < best_individual.calculate_fitness():
            best_individual = current_best
    
    return best_individual


if len(sys.argv) != 2:
    print("Loš unos, mora ovako: python gen.py input_file")
else:
    input_file = sys.argv[1]
    graph = read_graph(input_file)

    best_solution = None
    best_fitness = float('inf')
    total_fitness = 0

    for run in range(3):
        random.seed(run + int(time.time()))
        print(f"Rešenje {run+1}:")
        solution = evolutionary_algorithm(
            graph,
            population_size=150 if graph.n > 100 else 100,
            generations=300 if graph.n > 100 else 200,
            mutation_rate=0.2,
            tournament_size=7 if graph.n > 100 else 5
        )
        # u slučaju većeg grafa koriste se veće vrednosti parametara

        # ovo je jasno
        fitness = solution.calculate_fitness()
        total_fitness += fitness

        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = solution

        print(int(fitness) if fitness.is_integer() else fitness)
        sorted_edges = sorted(solution.edges)
        print(" ".join(f"({u} {v})" for u, v in sorted_edges))
        print()

        # sabijam ih u 3 txt fajla
        with open(f"res{run+1}.txt", "w") as f:
            f.write(f"{int(fitness) if fitness.is_integer() else fitness}\n")
            f.write(" ".join(f"({u} {v})" for u, v in sorted_edges) + "\n")

    # ispis na kraju
    avg_fitness = total_fitness / 3  # prosek iz ona 3 runova
    print(f"Prosečni fitness: {int(avg_fitness) if avg_fitness.is_integer() else avg_fitness}")
    print(f"Najbolji fitness: {int(best_fitness) if best_fitness.is_integer() else best_fitness}")
    sorted_edges = sorted(best_solution.edges)
    print(" ".join(f"({u} {v})" for u, v in sorted_edges)) # onaj format kao što stoji na drajvu