import pygame
import math
import random

class GA:
    def __init__(self):
        self.population = []
        self.points = []  # Make sure to define the 'points' variable

        # Define the following variables or adjust them as needed
        self.currentBest = None
        self.UNCHANGED_GENS = 0
        self.values = []
        self.best = None
        self.bestValue = None
        self.fitnessValues = []
        self.roulette = []
        self.dis = []
        self.mutationTimes = 0
        self.CROSSOVER_PROBABILITY = 0.5
        self.MUTATION_PROBABILITY = 0.5
        self.POPULATION_SIZE = 10

    def GAInitialize(self):
        self.countDistances()
        for i in range(self.POPULATION_SIZE):
            self.population.append(self.randomIndivial(len(self.points)))
        self.setBestValue()

    def GANextGeneration(self):
        self.currentGeneration += 1
        self.selection()
        self.crossover()
        self.mutation()
        self.setBestValue()

    def tribulate(self):
        for i in range(len(self.population) // 2, self.POPULATION_SIZE):
            self.population[i] = self.randomIndivial(len(self.points))

    def selection(self):
        parents = []
        initnum = 4
        parents.append(self.population[self.currentBest["bestPosition"]])
        parents.append(self.doMutate(self.best.clone()))
        parents.append(self.pushMutate(self.best.clone()))
        parents.append(self.best.clone())

        self.setRoulette()
        for i in range(initnum, self.POPULATION_SIZE):
            parents.append(self.population[self.wheelOut(random.random())])
        self.population = parents

    def crossover(self):
        queue = []
        for i in range(self.POPULATION_SIZE):
            if random.random() < self.CROSSOVER_PROBABILITY:
                queue.append(i)
        random.shuffle(queue)
        for i in range(0, len(queue) - 1, 2):
            self.doCrossover(queue[i], queue[i + 1])

    def doCrossover(self, x, y):
        child1 = self.getChild('next', x, y)
        child2 = self.getChild('previous', x, y)
        self.population[x] = child1
        self.population[y] = child2

    def getChild(self, fun, x, y):
        solution = []
        px = self.population[x].copy()
        py = self.population[y].copy()
        dx, dy = None, None
        c = px[self.randomNumber(len(px))]
        solution.append(c)
        while len(px) > 1:
            dx = px[fun](px.index(c))
            dy = py[fun](py.index(c))
            px.remove(c)
            py.remove(c)
            c = dx if self.dis[c][dx] < self.dis[c][dy] else dy
            solution.append(c)
        return solution

    def mutation(self):
        for i in range(self.POPULATION_SIZE):
            if random.random() < self.MUTATION_PROBABILITY:
                if random.random() > 0.5:
                    self.population[i] = self.pushMutate(self.population[i])
                else:
                    self.population[i] = self.doMutate(self.population[i])
                i -= 1

    def preciseMutate(self, orseq):
        seq = orseq.copy()
        if random.random() > 0.5:
            seq.reverse()
        bestv = self.evaluate(seq)
        for i in range(len(seq) // 2):
            for j in range(i + 2, len(seq) - 1):
                new_seq = self.swap_seq(seq, i, i + 1, j, j + 1)
                v = self.evaluate(new_seq)
                if v < bestv:
                    bestv = v
                    seq = new_seq
        return seq

    def preciseMutate1(self, orseq):
        seq = orseq.copy()
        bestv = self.evaluate(seq)
        for i in range(len(seq) - 1):
            new_seq = seq.copy()
            new_seq[i], new_seq[i + 1] = new_seq[i + 1], new_seq[i]
            v = self.evaluate(new_seq)
            if v < bestv:
                bestv = v
                seq = new_seq
        return seq

    def swap_seq(self, seq, p0, p1, q0, q1):
        seq1 = seq[:p0]
        seq2 = seq[p1 + 1:q1]
        seq2.append(seq[p0])
        seq2.append(seq[p1])
        seq3 = seq[q1:]
        return seq1 + seq2 + seq3

    def doMutate(self, seq):
        self.mutationTimes += 1
        while True:
            m = self.randomNumber(len(seq) - 2)
            n = self.randomNumber(len(seq))
            if m < n:
                break
        for i in range((n - m + 1) // 2):
            seq[m + i], seq[n - i] = seq[n - i], seq[m + i]
        return seq

    def pushMutate(self, seq):
        self.mutationTimes += 1
        m, n = None, None
        while True:
            m = self.randomNumber(len(seq) // 2)
            n = self.randomNumber(len(seq))
            if m < n:
                break
        s1 = seq[:m]
        s2 = seq[m:n]
        s3 = seq[n:]
        return s2 + s1 + s3.copy()

    def setBestValue(self):
        for i in range(len(self.population)):
            self.values[i] = self.evaluate(self.population[i])
        self.currentBest = self.getCurrentBest()
        if self.bestValue is None or self.bestValue > self.currentBest["bestValue"]:
            self.best = self.population[self.currentBest["bestPosition"]].copy()
            self.bestValue = self.currentBest["bestValue"]
            self.UNCHANGED_GENS = 0
        else:
            self.UNCHANGED_GENS += 1

    def getCurrentBest(self):
        bestP = 0
        currentBestValue = self.values[0]
        for i in range(1, len(self.population)):
            if self.values[i] < currentBestValue:
                currentBestValue = self.values[i]
                bestP = i
        return {
            "bestPosition": bestP,
            "bestValue": currentBestValue
        }

    def setRoulette(self):
        # Calculate all the fitness values
        for i in range(len(self.values)):
            self.fitnessValues[i] = 1.0 / self.values[i]
        # Set the roulette
        total = sum(self.fitnessValues)
        self.roulette = [0] * len(self.fitnessValues)
        for i in range(1, len(self.roulette)):
            self.roulette[i] = self.roulette[i - 1] + (self.fitnessValues[i] / total)

    def wheelOut(self, rand):
        for i in range(len(self.roulette)):
            if rand <= self.roulette[i]:
                return i

    def randomIndivial(self, n):
        a = list(range(n))
        random.shuffle(a)
        return a

    def evaluate(self, indivial):
        total_sum = self.dis[indivial[0]][indivial[-1]]
        for i in range(1, len(indivial)):
            total_sum += self.dis[indivial[i]][indivial[i - 1]]
        return total_sum

    def countDistances(self):
        length = len(self.points)
        self.dis = []
        for i in range(length):
            self.dis.append([0] * length)
            for j in range(length):
                self.dis[i][j] = self.distance(self.points[i], self.points[j])

    def distance(self, city1, city2):
        x1, y1 = city1
        x2, y2 = city2
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx**2 + dy**2)
        return distance
        #return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

    def randomNumber(self, n):
        return random.randint(0, n - 1)

#OLD DISTANCE Calculate the distance between two cities
#    def distance(city1, city2):
#        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


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
            ################################## best_route = solve_tsp()
            # Initialize the genetic algorithm
            ga.GAInitialize()
            for _ in range(num_generations):
                ga.GANextGeneration()

            # Retrieve the best solution found
            best_solution = ga.best
            best_value = ga.bestValue

        best_solution = 0
        best_value = 0
        # Draw text
        text = font.render("Press S to toggle solving TSP", True, BLACK)
        win.blit(text, (20, 20))
        text = font.render("Press R to reset", True, BLACK)
        win.blit(text, (20, 60))
        text = font.render("Best Solution:",best_solution, True, GREEN)
        win.blit(text, (20, 100))
        text = font.render("Best Value:",best_value, True, BLUE)
        win.blit(text, (20, 140))

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
    GREEN =(0, 255, 0)
    BLUE =(0, 0, 255)

    # Font
    font = pygame.font.Font(None, 30)

    # Number of cities
    num_cities = 30

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
    population_size = 30
    mutation_rate = 0.01
    num_generations = 1000  # Specify the number of generations you want to run

    ############### GA CLASS VARs  ####################################

    data40 = [(116, 404),(161, 617),(16, 97),(430, 536),(601, 504),(425, 461),(114, 544),(127, 118),(163, 357),(704, 104),(864, 125),(847, 523),(742, 170),(204, 601),(421, 377),(808, 49),(860, 466),(844, 294),(147, 213),(550, 124),(238, 313),(57, 572),(664, 190),(612, 644),(456, 154),(120, 477),(542, 313),(620, 29),(245, 246),(611, 578),(627, 373),(534, 286),(577, 545),(539, 340),(794, 328),(855, 139),(700, 47),(275, 593),(130, 196),(863, 35)];
    data200 = [(150, 172),(822, 244),(619, 220),(243, 433),(9, 48),(541, 402),(540, 212),(479, 646),(545, 90),(811, 355),(314, 325),(337, 487),(675, 76),(629, 375),(809, 105),(269, 135),(423, 592),(558, 288),(622, 70),(740, 495),(508, 79),(40, 236),(818, 252),(811, 480),(458, 220),(293, 220),(582, 275),(188, 542),(300, 235),(690, 649),(166, 565),(400, 80),(121, 498),(603, 587),(729, 89),(723, 23),(171, 609),(523, 449),(668, 102),(328, 531),(468, 588),(600, 239),(312, 636),(344, 112),(267, 184),(292, 615),(21, 401),(650, 266),(535, 393),(796, 598),(29, 412),(528, 363),(344, 152),(314, 35),(138, 191),(643, 341),(350, 423),(319, 542),(797, 659),(66, 296),(761, 574),(26, 270),(129, 509),(24, 312),(89, 635),(454, 34),(717, 189),(476, 457),(471, 212),(74, 457),(406, 221),(701, 313),(719, 642),(573, 424),(250, 231),(748, 334),(318, 453),(815, 92),(198, 47),(79, 451),(502, 582),(471, 355),(509, 257),(727, 290),(476, 281),(609, 576),(772, 72),(263, 156),(411, 203),(100, 254),(29, 208),(625, 349),(789, 163),(300, 224),(637, 57),(789, 153),(429, 427),(571, 355),(426, 348),(620, 545),(601, 322),(600, 441),(519, 357),(59, 262),(878, 621),(712, 592),(202, 341),(300, 41),(87, 647),(735, 60),(289, 110),(126, 133),(375, 584),(421, 469),(775, 341),(656, 534),(225, 634),(520, 339),(865, 515),(457, 378),(293, 141),(202, 293),(347, 423),(186, 284),(572, 600),(319, 412),(685, 73),(845, 248),(834, 339),(391, 571),(139, 346),(635, 352),(401, 117),(381, 281),(471, 552),(793, 585),(279, 520),(783, 520),(374, 38),(458, 479),(869, 15),(626, 216),(148, 604),(560, 109),(342, 141),(426, 536),(697, 414),(283, 18),(172, 181),(206, 227),(763, 291),(439, 124),(523, 388),(338, 211),(30, 593),(187, 498),(126, 86),(4, 58),(566, 329),(524, 486),(788, 334),(346, 194),(506, 231),(135, 190),(288, 406),(200, 515),(739, 91),(300, 439),(725, 420),(83, 612),(665, 336),(848, 246),(865, 521),(3, 406),(187, 431),(462, 564),(530, 648),(708, 173),(325, 96),(4, 480),(530, 512),(780, 126),(614, 610),(359, 431),(343, 640),(453, 182),(648, 477),(447, 258),(23, 465),(455, 215),(534, 396),(869, 337),(511, 290),(683, 291),(328, 370),(160, 497),(144, 203),(717, 222),(31, 376),(452, 600)];
    data500 = [(780, 560),(631, 173),(452, 237),(789, 506),(308, 175),(797, 157),(524, 583),(241, 7),(340, 105),(787, 19),(168, 342),(685, 386),(739, 195),(408, 550),(581, 577),(762, 406),(14, 370),(275, 610),(38, 484),(699, 148),(780, 272),(686, 611),(42, 650),(257, 329),(1, 260),(432, 448),(805, 546),(268, 472),(174, 154),(189, 432),(869, 653),(371, 337),(192, 279),(322, 118),(842, 584),(809, 381),(717, 250),(77, 575),(654, 21),(859, 146),(534, 561),(732, 227),(154, 371),(263, 148),(64, 524),(689, 553),(316, 358),(587, 374),(679, 125),(234, 501),(282, 403),(671, 107),(703, 347),(116, 408),(655, 593),(120, 196),(111, 240),(686, 271),(237, 213),(463, 562),(543, 240),(832, 406),(705, 280),(359, 252),(494, 575),(339, 85),(719, 115),(709, 564),(752, 178),(412, 599),(207, 524),(812, 359),(13, 500),(635, 477),(243, 236),(400, 381),(639, 551),(407, 65),(39, 619),(508, 170),(150, 115),(789, 353),(64, 178),(831, 434),(539, 83),(671, 317),(806, 479),(383, 335),(405, 103),(437, 549),(62, 590),(589, 296),(536, 539),(375, 541),(659, 326),(582, 600),(482, 73),(229, 8),(545, 292),(537, 174),(704, 273),(106, 487),(759, 575),(460, 358),(85, 6),(556, 112),(347, 196),(856, 88),(612, 395),(459, 195),(198, 431),(102, 14),(750, 403),(87, 37),(719, 146),(353, 405),(633, 476),(806, 313),(529, 509),(772, 55),(298, 527),(546, 522),(7, 72),(118, 337),(377, 216),(816, 327),(227, 167),(715, 422),(324, 516),(847, 170),(752, 422),(657, 570),(539, 450),(285, 556),(381, 168),(317, 251),(303, 197),(797, 50),(820, 193),(739, 85),(623, 118),(422, 73),(696, 205),(534, 450),(511, 263),(648, 110),(601, 518),(111, 627),(771, 572),(797, 303),(335, 332),(344, 492),(345, 610),(631, 340),(863, 305),(363, 406),(414, 14),(591, 26),(602, 592),(386, 273),(687, 183),(570, 27),(613, 645),(58, 268),(668, 375),(157, 349),(634, 627),(575, 465),(175, 460),(843, 625),(425, 20),(54, 411),(459, 659),(482, 176),(593, 296),(854, 512),(132, 551),(875, 577),(774, 470),(95, 584),(575, 614),(767, 635),(426, 212),(796, 38),(33, 147),(773, 95),(141, 640),(831, 257),(684, 175),(16, 534),(399, 579),(729, 185),(759, 217),(88, 327),(43, 167),(38, 161),(331, 405),(292, 130),(527, 658),(57, 288),(546, 479),(77, 118),(810, 74),(668, 101),(125, 570),(734, 267),(790, 417),(784, 204),(242, 335),(548, 458),(373, 189),(88, 216),(738, 1),(588, 384),(600, 221),(161, 340),(862, 400),(717, 82),(434, 19),(367, 476),(373, 288),(198, 508),(781, 516),(410, 401),(96, 377),(779, 653),(319, 404),(680, 66),(209, 381),(664, 41),(230, 340),(650, 499),(524, 604),(344, 287),(517, 351),(4, 10),(146, 233),(766, 185),(154, 476),(153, 534),(797, 278),(686, 434),(241, 469),(8, 550),(292, 118),(737, 118),(600, 610),(134, 405),(541, 96),(178, 53),(283, 618),(227, 559),(724, 264),(93, 192),(218, 531),(279, 395),(635, 430),(783, 424),(15, 34),(106, 406),(371, 277),(659, 222),(29, 401),(27, 194),(417, 657),(548, 12),(394, 160),(727, 410),(217, 459),(286, 629),(748, 105),(679, 514),(65, 487),(221, 160),(42, 239),(822, 390),(452, 291),(561, 107),(389, 451),(317, 94),(34, 50),(324, 284),(768, 531),(678, 432),(663, 411),(153, 27),(287, 348),(444, 184),(686, 482),(129, 122),(667, 368),(263, 78),(109, 190),(271, 208),(72, 346),(582, 5),(546, 343),(432, 305),(805, 5),(329, 100),(747, 304),(255, 283),(319, 623),(602, 145),(818, 582),(478, 491),(151, 451),(628, 605),(803, 260),(706, 636),(192, 535),(342, 177),(259, 599),(365, 229),(583, 426),(340, 562),(405, 629),(116, 260),(533, 479),(411, 615),(382, 125),(36, 272),(863, 466),(600, 288),(30, 648),(335, 269),(302, 92),(607, 98),(522, 101),(801, 339),(412, 189),(776, 446),(77, 528),(425, 547),(535, 317),(802, 229),(698, 534),(109, 109),(321, 37),(232, 115),(168, 621),(637, 502),(177, 156),(66, 376),(646, 329),(345, 290),(861, 28),(791, 184),(745, 244),(90, 370),(610, 617),(592, 452),(410, 500),(410, 288),(645, 239),(278, 163),(761, 27),(275, 33),(185, 203),(794, 129),(121, 421),(505, 126),(750, 309),(222, 518),(276, 272),(626, 61),(665, 320),(379, 38),(459, 357),(337, 450),(307, 418),(867, 631),(191, 272),(55, 465),(861, 291),(465, 101),(792, 81),(750, 278),(630, 488),(382, 539),(282, 527),(345, 575),(24, 421),(810, 491),(270, 356),(22, 646),(663, 617),(861, 452),(879, 409),(90, 515),(672, 416),(331, 68),(165, 570),(706, 384),(760, 85),(235, 477),(42, 451),(442, 598),(551, 539),(334, 419),(417, 656),(137, 610),(717, 505),(56, 619),(695, 527),(501, 514),(796, 315),(322, 218),(818, 215),(2, 239),(143, 232),(240, 38),(165, 277),(281, 91),(77, 297),(477, 18),(617, 407),(419, 170),(876, 275),(159, 277),(777, 104),(857, 25),(506, 418),(800, 170),(121, 625),(500, 579),(762, 294),(428, 614),(818, 584),(826, 101),(513, 566),(719, 638),(366, 121),(2, 142),(176, 382),(220, 280),(141, 210),(437, 419),(139, 84),(581, 449),(238, 485),(12, 139),(140, 324),(127, 542),(328, 314),(207, 123),(805, 285),(4, 566),(603, 592),(641, 77),(863, 498),(201, 387),(373, 357),(112, 322),(867, 472),(381, 633),(467, 234),(134, 63),(533, 468),(6, 185),(574, 362),(311, 451),(100, 572),(318, 47),(114, 650),(704, 641),(375, 355),(693, 391),(549, 154),(355, 167),(340, 493),(17, 98),(331, 179),(667, 431),(231, 460),(335, 270),(351, 0),(843, 449),(785, 1),(306, 86),(302, 496),(790, 236),(69, 49),(732, 160),(515, 73),(342, 253),(150, 579),(126, 317),(272, 432),(482, 301),(607, 622),(158, 53),(711, 480),(652, 193),(681, 151),(828, 359),(563, 71),(70, 138),(755, 192),(636, 133)];

    ga = GA()  # Create an instance of the GA class
    # Set the necessary parameters before calling the initialization method
    ga.points = data200  # Pass the list of points for evaluation
    ga.POPULATION_SIZE = population_size  # Set the population size
    ga.CROSSOVER_PROBABILITY = 0.8  # Set the crossover probability
    ga.MUTATION_PROBABILITY = mutation_rate  # Set the mutation probability
    ##################################################################
    main()
