import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import os
import tempfile
import itertools
os.environ["PATH"] += os.pathsep + 'C:/Users/billy/Downloads/graphviz-2.38/release/bin'

N, numNetworks, edges, numModules, t, trials, endTime  = 20, 1000, 40, 5, 2000, 2000, 0
timeseries = np.zeros((t, N))
rng = np.random.RandomState()
seeded_rng = np.random.RandomState(1729)

##test robustness of flipping a bit to landing in same basin
def get_random_modular(n, modules, directedEdges, p, getCommInfo=False, shared=None):
    pairings = {}
    assignments = np.zeros(n, dtype = int)
    for i in range(modules):
        pairings[i] = []
    adjMatrix = np.zeros((n,n))
    for i in range(n):
        randomModule = seeded_rng.randint(0, modules)
        pairings[randomModule].append(i)
        assignments[i] = randomModule

    def add_modular_edge(module = -1):
        if module == -1:
            randomComm = seeded_rng.randint(0, modules)
        else:
            randomComm = module
        while len(pairings[randomComm]) < 2:
            randomComm = seeded_rng.randint(0, modules)
        selection = seeded_rng.choice(pairings[randomComm], 2, replace=True)
        while adjMatrix[selection[0]][selection[1]] != 0:
            randomComm = seeded_rng.randint(0, modules)
            while len(pairings[randomComm]) < 2:
                randomComm = seeded_rng.randint(0, modules)
            selection = seeded_rng.choice(pairings[randomComm], 2, replace=True)
            #print("STUCK MODULAR")
        adjMatrix[selection[0]][selection[1]] += 1

    def add_random_edge(): #adds edge anywhere
        randEdge = seeded_rng.choice(n, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0:
            randEdge = seeded_rng.choice(n, 2, replace=False)
            #print("STUCK RANDOM")
        adjMatrix[randEdge[0]][randEdge[1]] += 1
    inModuleEdges = round(directedEdges * p)
    randEdges = directedEdges - inModuleEdges
    for i in range(inModuleEdges):
        #add_modular_edge(i % modules)
        add_modular_edge()
    for i in range(randEdges):
        add_random_edge()
    if shared is not None:
        shared.add(adjMatrix)
    if getCommInfo:
        return adjMatrix, pairings, assignments
    else:
        return adjMatrix


def getModularity(A, m, assignments):
    e_stored = np.zeros((len(assignments), len(assignments)))
    for r in range(len(A)):
        for c in range(len(A)):
            if A[r][c] != 0:
                e_stored[assignments[r]][int(assignments[c])] += 1
    e_stored /= m
    def a(i):
        return sum([e_stored[i][j] + e_stored[j][i] for j in range(len(A))])/2 -  e_stored[i][i]
    return sum([e_stored[i][i] - a(i) **2 for i in range (len(assignments))])


def getRandomTransitionFunctions():
    result = np.zeros((N, 2**N))
    for i in range(N):
        result[i] = seeded_rng.randint(0, 2, size = 2**N)
    return result

def getGraphInformedTransitionFunctions(adjMatrix):
    result = np.zeros((N, 2 **N))
    for i in range(N):
        print(i)
        col = adjMatrix[:, i]
        affectingIndices = np.nonzero(col)[0]
        randomOutputs = seeded_rng.randint(0, 2, size = 2 ** len(affectingIndices))
        # randomly maps the binary forms of 0 through len(affectingIndices) - 1 to an output
        for j in range(2 **N):
            state = convertToState(j)
            outputIndex = convertToIndex(state[affectingIndices])
            result[i][j] = randomOutputs[outputIndex]
    return result

def getGraphInformedTransitionFunctions2(adjMatrix):
    result = []
    for i in range(N):
        col = adjMatrix[:, i]
        affectingIndices = np.nonzero(col)[0]
        randomOutputs = seeded_rng.randint(0, 2, size = 2 ** len(affectingIndices))
        # randomly maps the binary forms of 0 through len(affectingIndices) - 1 to an output
        result.append((affectingIndices, randomOutputs))
    return result

def getRandomICs():
    return rng.randint(0, 2, size = N)

def convertToIndex(state):
    return int(state.dot(2**np.arange(state.size)[::-1]))

def convertToBinaryString(state):
    stringList = [str(int(x)) for x in state]
    return ''.join(stringList)

def convertToStateFromString(stringState):
    listState = list(stringState)
    listState = [int(x) for x in listState]
    return np.array(listState)

def convertToState(index):
    stringState = bin(index)[2:].zfill(N)
    L = list(stringState)
    return np.array([int(x) for x in L])

def timeStep(funcs, state):
    index = convertToIndex(state)
    newState = np.zeros(N)
    for i in range(len(newState)):
        newState[i] = funcs[i][index]
    return newState

def timeStep2(funcs, state):
    newState = np.zeros(N)
    for i in range(len(newState)):
        affectingIndices = funcs[i][0]
        outputVal = funcs[i][1][convertToIndex(state[affectingIndices])]
        newState[i] = outputVal
    return newState

def runSim(ICs, funcs):
    global timeseries
    global endTime
    seenStates = []
    timeseries = np.zeros((t, N))
    state = ICs
    for i in range(t):
        timeseries[i] = state
        seenStates.append(convertToIndex(state))
        state = timeStep2(funcs, state)
        if seenStates.__contains__(convertToIndex(state)):
            print("yay " + str(i))
            if i != t - 1:
                timeseries[i+1] = state
                endTime = i + 1
            break

def runSim2(ICs, funcs, basins, basin_assignments, attractor_sizes, allowBasinAddition = True):
    global timeseries
    global endTime
    seenStates = dict()
    state = ICs
    attractorSize = 0
    fixation_time = 0
    for i in range(t):
        timeseries[i] = state
        seenStates[convertToBinaryString(state)] = i
        state = timeStep2(funcs, state)
        stringState = convertToBinaryString(state)
        if stringState in seenStates.keys():
            if i != t - 1:
                timeseries[i+1] = state
                endTime = i + 1
                attractorSize = 1 + i - seenStates[stringState]
                fixation_time = i
            break
    for val in basins:
        if val in seenStates:
            for seen in seenStates:
                basin_assignments[seen] = val
            return fixation_time
    if allowBasinAddition:
        basin_string = convertToBinaryString(state)
        basins.add(basin_string)
        attractor_sizes[basin_string] = attractorSize
        for seen in seenStates:
            basin_assignments[seen] = basin_string
    return fixation_time


def buildPartialTransitionNetwork2(network, funcs, neglectNetwork = False):
    basins = set()
    basin_assignments = {}
    attractor_sizes = {}
    fixation_times = []
    for i in range(trials):
        randomIC = rng.randint(0, 2 **N, dtype = np.int64)
        stateIC = convertToState(randomIC)
        fixation_time = runSim2(stateIC, funcs, basins, basin_assignments, attractor_sizes)
        fixation_times.append(fixation_time)
        if not neglectNetwork:
            createTransitionNetwork(network)
    return basins, basin_assignments, attractor_sizes, fixation_times

def buildFullTransitionNetwork2(network, funcs, neglectNetwork = False):
    basins = set()
    basin_assignments = {}
    attractor_sizes = {}
    for i in range(2 **N):
        print(i)
        IC = convertToState(i)
        runSim2(IC, funcs, basins, basin_assignments, attractor_sizes)
        if not neglectNetwork:
            createTransitionNetwork(network)
    return basins, basin_assignments

def buildInfluenceDigraph(adjMatrix, assignments):
    result = Digraph(strict = True)
    for i in range(N):
        result.node(str(i), str(i) + "_" + str(int(assignments[i])))
    for r in range(N):
        for c in range(N):
            if adjMatrix[r][c] == 1:
                result.edge(str(r), str(c))
    return result

def createTransitionNetwork(network):
    nodes = []
    edges = []
    for i in range(endTime):
        stringNode = convertToBinaryString(timeseries[i])
        stringNext = convertToBinaryString(timeseries[i + 1])
        if not nodes.__contains__(stringNode):
            nodes.append(stringNode)
        edges.append((stringNode, stringNext))

    for i in range(len(nodes)):
        network.node(nodes[i])
    for i in range(len(edges)):
        network.edge(edges[i][0], edges[i][1])

def buildFullTransitionNetwork(network, funcs):
    for i in range(2 **N):
        print(i)
        IC = convertToState(i)
        runSim(IC, funcs)
        createTransitionNetwork(network)

def buildPartialTransitionNetwork(network, funcs):
    for i in range(trials):
        print(i)
        randomIC = rng.randint(0, 2 **N, dtype = np.int64)
        stateIC = convertToState(randomIC)
        runSim(stateIC, funcs)
        createTransitionNetwork(network)

def get_corresponding_basin(funcs, basins, basin_assignments, attractor_sizes, state):
    stringState = convertToBinaryString(state)
    if stringState in basin_assignments.keys():
        return basin_assignments[stringState]
    runSim2(state, funcs, basins, basin_assignments,attractor_sizes, allowBasinAddition = True)
    return basin_assignments[stringState]


def get_basin_attributes(funcs, basins, basin_assignments, attractor_sizes, probes):
    basin_counts = {}
    num_basins = len(basins)
    largestBasinSize = 0
    avg_attractor_length = 0
    num_detected = len(basin_assignments)
    attractor_size_values = []
    for val in basins:
        basin_counts[val] = 0
    for key in basin_assignments.keys():
        basin_counts[basin_assignments[key]] += 1
    for key in basin_counts.keys():
        if basin_counts[key] > largestBasinSize:
            largestBasinSize = basin_counts[key]
    for key in attractor_sizes.keys():
        avg_attractor_length += attractor_sizes[key]
        attractor_size_values.append(attractor_sizes[key])
    stableCount = 0
    for i in range(probes):
        state = getRandomICs()
        originalBasin = get_corresponding_basin(funcs, basins, basin_assignments,attractor_sizes, state)
        randIndex = rng.randint(0, len(state))
        if state[randIndex] == 0:
            state[randIndex] = 1
        else:
            state[randIndex] = 0
        if originalBasin == get_corresponding_basin(funcs, basins, basin_assignments,attractor_sizes, state):
            stableCount += 1
    return basin_counts, num_basins, largestBasinSize/num_detected, stableCount/probes, avg_attractor_length/len(attractor_sizes), attractor_size_values




