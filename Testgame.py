import pygame
import random
import math

# Constants
POPULATION_SIZE = 100
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2

# Pygame initialization
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# City coordinates
cities = [
    (400, 50),
    (600, 300),
    (200, 400),
    (600, 500),
    (100, 200)
]

# Helper functions
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def evaluate(individual):
    length = len(individual)
    total_distance = distance(cities[individual[0]], cities[individual[length - 1]])
    for i in range(1, length):
        total_distance += distance(cities[individual[i]], cities[individual[i-1]])
    return total_distance

def random_individual(n):
    individual = list(range(n))
    random.shuffle(individual)
    return individual

def crossover(x, y):
    child1 = getChild('next', x, y)
    child2 = getChild('previous', x, y)
    population[x] = child1
    population[y] = child2

def getChild(fun, x, y):
    solution = []
    px = population[x].copy()
    py = population[y].copy()
    c = px[random.randint(0, len(px) - 1)]
    solution.append(c)
    while len(px) > 1:
        dx = px[px.index(c) + 1] if fun == 'next' else px[px.index(c) - 1]
        dy = py[py.index(c) + 1] if fun == 'next' else py[py.index(c) - 1]
        px.remove(c)
        py.remove(c)
        c = dx if distance[c][dx] < distance[c][dy] else dy
        solution.append(c)
    return solution

def mutation():
    for i in range(POPULATION_SIZE):
        if random.random() < MUTATION_PROBABILITY:
            if random.random() > 0.5:
                population[i] = push_mutate(population[i])
            else:
                population[i] = do_mutate(population[i])

def do_mutate(seq):
    mutation_times = 0
    m, n = 0, 0
    while m >= n:
        m = random.randint(0, len(seq) - 3)
        n = random.randint(m + 2, len(seq) - 1)
    for i in range((n - m + 1) // 2):
        seq[m + i], seq[n - i] = seq[n - i], seq[m + i]
    return seq

def push_mutate(seq):
    mutation_times = 0
    m, n = 0, 0
    while m >= n:
        m = random.randint(0, len(seq) // 2 - 1)
        n = random.randint(m + 1, len(seq) - 1)
    s1 = seq[:m]
    s2 = seq[m:n]
    s3 = seq[n:]
    return s2 + s1 + s3

def initialize_population():
    population = []
    count_distances()
    for _ in range(POPULATION_SIZE):
        population.append(random_individual(len(cities)))
    set_best_value()
    return population

def set_best_value():
    for i in range(len(population)):
        values[i] = evaluate(population[i])
    best_position, best_value = get_current_best()
    if 'best' not in globals() or best_value < globals()['bestValue']:
        globals()['best'] = population[best_position].copy()
        globals()['bestValue'] = best_value
        globals()['UNCHANGED_GENS'] = 0
    else:
        globals()['UNCHANGED_GENS'] += 1

def get_current_best():
    best_position = 0
    current_best_value = values[0]
    for i in range(1, len(population)):
        if values[i] < current_best_value:
            current_best_value = values[i]
            best_position = i
    return best_position, current_best_value

def count_distances():
    length = len(cities)
    distance_matrix = [[0] * length for _ in range(length)]
    for i in range(length):
        for j in range(length):
            distance_matrix[i][j] = math.floor(distance(cities[i], cities[j]))
    globals()['distance'] = distance_matrix

# Initialize the genetic algorithm
population = initialize_population()
values = [0] * POPULATION_SIZE

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Genetic algorithm operations
    set_best_value()
    selection()
    crossover()
    mutation()

    # Update the display
    screen.fill((255, 255, 255))
    for i, city in enumerate(cities):
        pygame.draw.circle(screen, (0, 0, 0), city, 5)
        if i > 0:
            pygame.draw.line(screen, (0, 0, 0), cities[best[i - 1]], city)
    pygame.draw.line(screen, (0, 0, 0), cities[best[-1]], cities[0])
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
