import pygame
import math
import random

# Calculate the distance between two cities
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Calculate the total distance of a given route
def calculate_distance(route):
    total_distance = 0
    for i in range(len(route)):
        city1 = cities[route[i]]
        city2 = cities[route[(i + 1) % len(route)]]
        total_distance += distance(city1, city2)
    return total_distance

# Calculate the total size of a given route
def calculate_size(route):
    total_size = 0
    for i in range(len(route)):
        city = cities[route[i]]
        total_size += city[2]
    return total_size

# Generate a random population of routes
def generate_population():
    population = []
    for _ in range(population_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

# Perform selection based on fitness (lower distance is better)
def selection(population):
    fitness_scores = [1 / calculate_distance(route) for route in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected = random.choices(population, probabilities, k=2)
    return selected[0], selected[1]

# Perform crossover between two routes using ordered crossover (OX)
def crossover(parent1, parent2):
    start = random.randint(0, num_cities - 1)
    end = random.randint(start + 1, num_cities)
    child = [-1] * num_cities
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child]
    j = 0
    for i in range(num_cities):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    return child

# Perform mutation on a route by swapping two cities
def mutate(route):
    if random.random() < mutation_rate:
        index1, index2 = random.sample(range(num_cities), 2)
        route[index1], route[index2] = route[index2], route[index1]
    return route

# Solve the TSP using Genetic Algorithm
def solve_tsp():
    population = generate_population()
    best_distance = float("inf")
    best_route = None

    for _ in range(num_generations):
        next_generation = []

        for _ in range(population_size // 2):
            parent1, parent2 = selection(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation.extend([child1, child2])

        population = next_generation

        for route in population:
            distance = calculate_distance(route)
            if distance < best_distance:
                best_distance = distance
                best_route = route

    return best_route

# Main game loop
def main():
    dragging = False
    running = True
    solve_tsp_flag = False  # Flag to indicate if the TSP should be solved
    solve_tsp_toggle = False  # Toggle flag for solving TSP
    generation = 0  # Current generation counter
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), (random.randint(0,255), random.randint(0,255), random.randint(0,255))]  # Colors for drawing lines in each generation

    while running:
        win.fill(WHITE)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    solve_tsp_toggle = not solve_tsp_toggle  # Toggle the solve TSP flag
                    if solve_tsp_toggle:
                        generation += 1  # Increment generation counter
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_s:
                    solve_tsp_flag = solve_tsp_toggle  # Set the solve TSP flag
                elif event.key == pygame.K_r:
                    cities.clear()
                    for _ in range(num_cities):
                        x = random.randint(50, WIDTH - 50)
                        y = random.randint(50, HEIGHT - 50)
                        size = random.randint(1, 10)
                        cities.append((x, y, size))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for i, city in enumerate(cities):
                        if pygame.Rect(city[0] - 10, city[1] - 10, 20, 20).collidepoint(event.pos):
                            dragging = True
                            selected_city = i
                            mouse_x, mouse_y = event.pos
                            offset = (city[0] - mouse_x, city[1] - mouse_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_x, mouse_y = event.pos
                    cities[selected_city] = (mouse_x + offset[0], mouse_y + offset[1], cities[selected_city][2])

        # Draw cities and transportable goods size
        for city in cities:
            pygame.draw.circle(win, BLACK, city[:2], 5)
            text = font.render(str(city[2]), True, BLACK)
            win.blit(text, (city[0] + 10, city[1] + 10))

        # Draw connections between cities
        for i in range(len(cities)):
            city1 = cities[i]
            city2 = cities[(i + 1) % len(cities)]
            pygame.draw.line(win, colors[generation % len(colors)], city1[:2], city2[:2], 2)

        # Solve TSP if the flag is set
        if solve_tsp_flag:
            best_route = solve_tsp()
            if best_route:
                for i in range(len(best_route)):
                    city1 = cities[best_route[i]]
                    city2 = cities[best_route[(i + 1) % len(best_route)]]
                    pygame.draw.line(win, RED, city1[:2], city2[:2], 3)

        # Draw text
        text = font.render("Press S to toggle solving TSP", True, BLACK)
        win.blit(text, (20, 20))
        text = font.render("Press R to reset", True, BLACK)
        win.blit(text, (20, 60))

        pygame.display.flip()

    pygame.quit()

# Start the program
if __name__ == "__main__":
    # Pygame initialization
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Travelling Salesman Problem Solver")

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)

    # Font
    font = pygame.font.Font(None, 30)

    # Number of cities
    num_cities = 10

    # Variables for dragging cities
    selected_city = None
    offset = (0, 0)

    # Generate random city positions and transportable goods size
    cities = []
    for _ in range(num_cities):
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        size = random.randint(1, 10)
        cities.append((x, y, size))

    # GA parameters
    population_size = 10
    mutation_rate = 0.01
    num_generations = 100
    main()
