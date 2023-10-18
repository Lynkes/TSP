import random
import copy

POPULATION_SIZE = 100
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.01
UNCHANGED_GENS = 0

population = []
points = []  # Add your points data here

import random

def GAInitialize():
    countDistances()
    for i in range(POPULATION_SIZE):
        population.append(randomIndivial(len(points)))
    setBestValue()

def GANextGeneration():
    global currentGeneration
    currentGeneration += 1
    selection()
    crossover()
    mutation()
    setBestValue()

def tribulate():
    for i in range(len(population)//2, POPULATION_SIZE):
        population[i] = randomIndivial(len(points))

def selection():
    parents = []
    initnum = 4
    parents.append(population[currentBest["bestPosition"]])
    parents.append(doMutate(best.clone()))
    parents.append(pushMutate(best.clone()))
    parents.append(best.clone())

    setRoulette()
    for i in range(initnum, POPULATION_SIZE):
        parents.append(population[wheelOut(random.random())])
    population = parents

def crossover():
    queue = []
    for i in range(POPULATION_SIZE):
        if random.random() < CROSSOVER_PROBABILITY:
            queue.append(i)
    random.shuffle(queue)
    for i in range(0, len(queue)-1, 2):
        doCrossover(queue[i], queue[i+1])

def doCrossover(x, y):
    child1 = getChild('next', x, y)
    child2 = getChild('previous', x, y)
    population[x] = child1
    population[y] = child2

def getChild(fun, x, y):
    solution = []
    px = population[x].clone()
    py = population[y].clone()
    dx, dy = None, None
    c = px[randomNumber(len(px))]
    solution.append(c)
    while len(px) > 1:
        dx = px[fun](px.index(c))
        dy = py[fun](py.index(c))
        px.remove(c)
        py.remove(c)
        c = dx if dis[c][dx] < dis[c][dy] else dy
        solution.append(c)
    return solution

def mutation():
    for i in range(POPULATION_SIZE):
        if random.random() < MUTATION_PROBABILITY:
            if random.random() > 0.5:
                population[i] = pushMutate(population[i])
            else:
                population[i] = doMutate(population[i])
            i -= 1

def preciseMutate(orseq):
    seq = orseq.copy()
    if random.random() > 0.5:
        seq.reverse()
    bestv = evaluate(seq)
    for i in range(len(seq)//2):
        for j in range(i+2, len(seq)-1):
            new_seq = swap_seq(seq, i, i+1, j, j+1)
            v = evaluate(new_seq)
            if v < bestv:
                bestv = v
                seq = new_seq
    return seq

def preciseMutate1(orseq):
    seq = orseq.copy()
    bestv = evaluate(seq)
    for i in range(len(seq)-1):
        new_seq = seq.copy()
        new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
        v = evaluate(new_seq)
        if v < bestv:
            bestv = v
            seq = new_seq
    return seq

def swap_seq(seq, p0, p1, q0, q1):
    seq1 = seq[:p0]
    seq2 = seq[p1+1:q1]
    seq2.append(seq[p0])
    seq2.append(seq[p1])
    seq3 = seq[q1:]
    return seq1 + seq2 + seq3

def doMutate(seq):
    mutationTimes += 1
    while True:
        m = randomNumber(len(seq) - 2)
        n = randomNumber(len(seq))
        if m < n:
            break
    for i in range((n - m + 1)//2):
        seq[m+i], seq[n-i] = seq[n-i], seq[m+i]
    return seq

def pushMutate(seq):
    mutationTimes += 1
    m, n = None, None
    while True:
        m = randomNumber(len(seq)//2)
        n = randomNumber(len(seq))
        if m < n:
            break
    s1 = seq[:m]
    s2 = seq[m:n]
    s3 = seq[n:]
    return s2 + s1 + s3.copy()

def setBestValue():
    for i in range(len(population)):
        values[i] = evaluate(population[i])
    currentBest = getCurrentBest()
    if bestValue is None or bestValue > currentBest["bestValue"]:
        best = population[currentBest["bestPosition"]].copy()
        bestValue = currentBest["bestValue"]
        UNCHANGED_GENS = 0
    else:
        UNCHANGED_GENS += 1

def getCurrentBest():
    bestP = 0
    currentBestValue = values[0]
    for i in range(1, len(population)):
        if values[i] < currentBestValue:
            currentBestValue = values[i]
            bestP = i
    return {
        "bestPosition": bestP,
        "bestValue": currentBestValue
    }

def setRoulette():
    # calculate all the fitness
    for i in range(len(values)):
        fitnessValues[i] = 1.0 / values[i]
    # set the roulette
    sum = 0
    for i in range(len(fitnessValues)):
        sum += fitnessValues[i]
    for i in range(1, len(roulette)):
        roulette[i] += roulette[i-1]

def wheelOut(rand):
    for i in range(len(roulette)):
        if rand <= roulette[i]:
            return i

def randomIndivial(n):
    a = list(range(n))
    random.shuffle(a)
    return a

def evaluate(indivial):
    sum = dis[indivial[0]][indivial[-1]]
    for i in range(1, len(indivial)):
        sum += dis[indivial[i]][indivial[i-1]]
    return sum

def countDistances():
    length = len(points)
    dis = []
    for i in range(length):
        dis.append([0] * length)
        for j in range(length):
            dis[i][j] = distance(points[i], points[j])

