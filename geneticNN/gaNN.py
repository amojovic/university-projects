import random
import copy
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

POP_SIZE            = 10
GENERATIONS         = 5
ELITE_COUNT         = 2
TOURNAMENT_K        = 3
MUTATION_PROB       = 0.2
TRAIN_EPOCHS        = 2
FINAL_EPOCHS        = 5
SURROGATE_RATIO     = 0.2
SUR_WINDOW          = 20
SUR_STAB_THRESH     = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on: {DEVICE}")

def tournament_select(indices, scores, k):
    k = min(k, len(indices))
    aspirants = random.sample(indices, k)
    return max(aspirants, key=lambda i: scores[i])

def random_genome():
    n = random.choice([2, 3])
    return {
        'n_conv': n,
        'channels': [random.choice([16, 32, 48, 64, 128]) for _ in range(n)],
        'kernel_sizes': [random.choice([3, 5, 7]) for _ in range(n)],
        'dropout': random.uniform(0.0, 0.5),
        'lr': 10**random.uniform(-4, -2),
        'activation': random.choice(['relu', 'tanh', 'elu', 'sigmoid'])
    }

class EvolvedCNN(nn.Module):
    def __init__(self, genome):
        super().__init__()
        act_map = {
            'relu':    nn.ReLU(inplace=True),
            'tanh':    nn.Tanh(),
            'elu':     nn.ELU(inplace=True),
            'sigmoid': nn.Sigmoid()
        }
        act_layer = act_map[genome['activation']]
        layers = []
        in_ch = 1
        for out_ch, k in zip(genome['channels'], genome['kernel_sizes']):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2),
                act_layer,
                nn.MaxPool2d(2)
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        dummy = torch.zeros(1, 1, 28, 28)
        flat_size = self.conv(dummy).view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Dropout(genome['dropout']),
            nn.Linear(flat_size, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def fitness(genome, train_loader, val_loader):
    model = EvolvedCNN(genome).to(DEVICE)
    if 'weights' in genome:
        try:
            model.load_state_dict(genome['weights'])
        except RuntimeError:
            pass
    optimizer = optim.Adam(model.parameters(), lr=genome['lr'])
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(TRAIN_EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    genome['weights'] = copy.deepcopy(model.state_dict())
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

def genome_distance(g1, g2):
    d = abs(g1['n_conv'] - g2['n_conv'])
    ch1 = g1['channels'] + [0]*(3 - len(g1['channels']))
    ch2 = g2['channels'] + [0]*(3 - len(g2['channels']))
    d += sum(abs(a - b)/128 for a, b in zip(ch1, ch2))
    k1 = g1['kernel_sizes'] + [0]*(3 - len(g1['kernel_sizes']))
    k2 = g2['kernel_sizes'] + [0]*(3 - len(g2['kernel_sizes']))
    d += sum(abs(a - b)/7 for a, b in zip(k1, k2))
    d += 0 if g1['activation'] == g2['activation'] else 1
    return d

def speciate(pop):
    species = []
    for idx, g in enumerate(pop):
        for sp in species:
            if genome_distance(pop[sp[0]], g) < 1.5:
                sp.append(idx)
                break
        else:
            species.append([idx])
    return species

def crossover(p1, p2):
    child = {}
    for key in ['n_conv', 'channels', 'kernel_sizes', 'dropout', 'lr', 'activation']:
        child[key] = copy.deepcopy(random.choice([p1[key], p2[key]]))
    base = random.choice([p1, p2])
    if 'weights' in base:
        child['weights'] = copy.deepcopy(base['weights'])
    return child

def mutate(genome):
    structural_change = False
    if random.random() < MUTATION_PROB:
        genome['n_conv'] = random.choice([2, 3])
        structural_change = True
    if random.random() < MUTATION_PROB:
        genome['activation'] = random.choice(['relu', 'tanh', 'elu', 'sigmoid'])
    n = genome['n_conv']
    genome['channels'] = (genome['channels'][:n] +
                          [random.choice([16,32,48,64,128]) for _ in range(n - len(genome['channels']))])
    genome['kernel_sizes'] = (genome['kernel_sizes'][:n] +
                              [random.choice([3,5,7]) for _ in range(n - len(genome['kernel_sizes']))])
    for i in range(n):
        if random.random() < MUTATION_PROB:
            genome['channels'][i] = random.choice([16,32,48,64,128])
            structural_change = True
        if random.random() < MUTATION_PROB:
            genome['kernel_sizes'][i] = random.choice([3,5,7])
            structural_change = True
    if random.random() < MUTATION_PROB:
        genome['dropout'] = random.uniform(0.0, 0.5)
    if random.random() < MUTATION_PROB:
        genome['lr'] = 10**random.uniform(-4, -2)
    if structural_change and 'weights' in genome:
        genome.pop('weights')
    return genome

def encode_genome(genome):
    vec = [genome['n_conv']]
    ch = genome['channels'] + [0]*(3 - len(genome['channels']))
    ks = genome['kernel_sizes'] + [0]*(3 - len(genome['kernel_sizes']))
    act = {'relu': 0, 'tanh': 1, 'elu': 2, 'sigmoid': 3}[genome['activation']]
    vec += ch + ks + [genome['dropout'], genome['lr'], act]
    return np.array(vec, dtype=float)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train = datasets.MNIST('.', train=True, download=True, transform=transform)
train_ds, val_ds = random_split(full_train, [50000, 10000])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
test_ds    = datasets.MNIST('.', train=False, download=True, transform=transform)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

population = [random_genome() for _ in range(POP_SIZE)]
fitnesses  = [fitness(g, train_loader, val_loader) for g in population]
sur_X, sur_y = []
surrogate_on = False

for gen in range(1, GENERATIONS+1):
    print(f"\n--- generation {gen} ---")
    species = speciate(population)
    elite_idx = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)[:ELITE_COUNT]
    new_pop = [population[i] for i in elite_idx]
    new_fit = [fitnesses[i] for i in elite_idx]
    offspring = []
    while len(offspring) < POP_SIZE - ELITE_COUNT:
        sp = random.choices(species, weights=[len(s) for s in species])[0]
        i1 = tournament_select(sp, fitnesses, TOURNAMENT_K)
        i2 = tournament_select(sp, fitnesses, TOURNAMENT_K)
        offspring.append(mutate(crossover(population[i1], population[i2])))
    X_off = np.vstack([encode_genome(c) for c in offspring])
    if surrogate_on and len(sur_y) >= SUR_WINDOW:
        rf = RandomForestRegressor()
        rf.fit(np.vstack(sur_X), np.array(sur_y))
        preds = rf.predict(X_off)
        k = max(1, int(SURROGATE_RATIO * len(offspring)))
        real_idx = np.argsort(preds)[-k:]
    else:
        preds = np.zeros(len(offspring))
        real_idx = np.arange(len(offspring))
    off_fitness = np.zeros(len(offspring))
    for i in real_idx:
        acc = fitness(offspring[i], train_loader, val_loader)
        off_fitness[i] = acc
        sur_X.append(X_off[i])
        sur_y.append(acc)
    if surrogate_on:
        for i in set(range(len(offspring))) - set(real_idx):
            off_fitness[i] = preds[i]
    best_idxs = np.argsort(off_fitness)[- (POP_SIZE - ELITE_COUNT):]
    for i in best_idxs:
        new_pop.append(offspring[i])
        new_fit.append(off_fitness[i])
    population = new_pop
    fitnesses  = new_fit
    print(f" best validation acc this gen: {max(fitnesses):.4f}")
    if len(sur_y) >= SUR_WINDOW:
        rf_stab = RandomForestRegressor()
        rf_stab.fit(np.vstack(sur_X), np.array(sur_y))
        X_last = np.vstack(sur_X[-SUR_WINDOW:])
        y_last = np.array(sur_y[-SUR_WINDOW:])
        preds_last = rf_stab.predict(X_last)
        corr = np.corrcoef(preds_last, y_last)[0,1]
        surrogate_on = (corr >= SUR_STAB_THRESH)
        print(f" surrogate stability corr={corr:.2f} → {'on' if surrogate_on else 'off'}")
    else:
        print(f" surrogate disabled: only {len(sur_y)} real samples (< {SUR_WINDOW})")

best_idx = int(np.argmax(fitnesses))
best_genome = population[best_idx]
print("\nbest genome:", best_genome)

model = EvolvedCNN(best_genome).to(DEVICE)
if 'weights' in best_genome:
    try:
        model.load_state_dict(best_genome['weights'])
    except RuntimeError:
        pass

opt = optim.Adam(model.parameters(), lr=best_genome['lr'])
loss_fn = nn.CrossEntropyLoss()
for _ in range(FINAL_EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

model.eval()
correct = total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
print(f"test accuracy: {correct/total:.4f}")
