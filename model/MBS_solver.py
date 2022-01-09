import sys
import time
import math
import pickle
import copy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from itertools import combinations_with_replacement, product, permutations
from discreteMarkovChain import markovChain
from sympy import *
from sympy.solvers.solveset import linsolve
from sympy.solvers import solve
from bayes_opt import BayesianOptimization
import threading
# from oct2py import Oct2Py
import multiprocessing
from multiprocessing import Process
# from oct2py import Oct2Py
#from butools.map import *
#%run "~/github/butools/Python/BuToolsInit.py"
#%run '/home/rpinciro/butools/Python/BuToolsInit'
from butools.map import *
#import butools
#from butools import *
butools.verbose = True
butools.checkInput = False

dict_size = {
        0 : 11, 1 : 22.1, 2 : 25.7, 3 : 30.4, 4 : 33.5,
        5 : 41.9, 6 : 43.9, 7 : 44.7, 8 : 51.9, 9 : 56,
        10 : 59.4, 11 : 63.5, 12 : 67.5, 13 : 68.6, 14 : 76,
        15 : 79, 16 : 81.6, 17 : 87.9, 18 : 91.9, 19 : 96,
        20 : 99.8, 21 : 103.5, 22 : 106.9, 23 : 110.8,
        24 : 115.9, 25 : 118.7, 26 : 123.8, 27 : 127.5,
        28 : 131, 29 : 134.5, 30 : 139.9, 31 : 143.5 }

class BayesOpt:
    
    memMinIncrement = 256
    
    def minimizeCost(self, bs, to, mem):
        mem = int(mem) * self.memMinIncrement
        begin_time = time.time()
        latCdf = latencyCdfMulticlass(self.D0, self.D1, int(bs), to, self.Kcl, self.Kprob, self.W, mem=mem, pi=self.pi, addTimeout=True)
        #print("done meauring latency\t{}".format(time.time()-begin_time))
        percentile_time = time.time()
        percLat = getQuantile(mergeClassValues(latCdf, self.Kprob), self.quantile)
        #print("Done calculating percentile latency\t{}".format(time.time()-percentile_time))
        cost_time = time.time()
        #print(percLat)
        if percLat < self.slo:
            costCdf = costCdfMulticlass(self.D0, self.D1, int(bs), to, self.Kcl, self.Kprob, self.W, mem=mem, pi=self.pi)
            #print("Done calculating cost\t{}".format(time.time()-cost_time))
            #input("Press enter for next iteration")
            return -1 * getMeanValue(mergeClassValues(costCdf, self.Kprob))
        else:
            return -10**9
        
    def minimizeLatency(self, bs, to, mem):
        
        mem = int(mem) * self.memMinIncrement
        costCdf = costCdfMulticlass(self.D0, self.D1, int(bs), to, self.Kcl, self.Kprob, self.W, mem=mem, pi=self.pi, addTimeout=True)
        percCost = getQuantile(mergeClassValues(costCdf, self.Kprob), self.quantile)
        if percCost < self.slo:
            latCdf = latencyCdfMulticlass(self.D0, self.D1, int(bs), to, self.Kcl, self.Kprob, self.W, mem=mem, pi=self.pi)
            return -1 * getMeanValue(mergeClassValues(latCdf, self.Kprob))
        else:
            return -10**9
        
    def optimize(self, verbose=False):
        pbounds = {'bs': (self.Bmin, self.Bmax), 'to': (self.Tmin, self.Tmax), 'mem': (self.Mmin, self.Mmax)}

        start = time.time()
        if self.constraint == 'latency':
            optimizer = BayesianOptimization(f=self.minimizeCost, pbounds=pbounds, random_state=123, verbose=0)
        elif self.constraint == 'cost':
            optimizer = BayesianOptimization(f=self.minimizeLatency, pbounds=pbounds, random_state=123, verbose=0)
        else:
            print('This constraint is not supported.')
            raise stopExecution
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
        end = time.time()
        output = (int(optimizer.max['params']['mem'])*self.memMinIncrement, optimizer.max['params']['to'], int(optimizer.max['params']['bs']), end-start, -optimizer.max['target'])
        print(output)
        with open("temp_config", 'a+') as fp:
            fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(self.num_buffs, self.buff_id,int(optimizer.max['params']['mem'])*self.memMinIncrement, \
             optimizer.max['params']['to'], int(optimizer.max['params']['bs']), float(-optimizer.max['target'])))
        #return output
        
    def __init__(self, D0, D1, pi, Kcl, Kprob, W, quantile, slo, constraint, Bmin, Bmax, Tmin, Tmax, Mmin, Mmax, init_points, n_iter, num_buffs, buff_id):
        self.D0 = D0
        self.D1 = D1
        self.pi = pi
        self.Kcl = Kcl
        self.Kprob = Kprob
        self.W = W
        self.quantile = quantile
        self.slo = slo
        self.constraint = constraint
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Mmin = Mmin / self.memMinIncrement
        self.Mmax = Mmax / self.memMinIncrement
        self.init_points = init_points
        self.n_iter = n_iter
        self.num_buffs = num_buffs
        self.buff_id = buff_id


# This function save an object to a pickle file
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# This function load an object from a pickle file
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
    iteration   - Required  : current iteration (Int)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    length      - Optional  : character length of bar (Int)
    fill        - Optional  : bar fill character (Str)
    printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print("This is replacement")
    #print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



class StopExecution(Exception):
    def _render_traceback_(self):
        pass

    
def mergeLists(samplesPerClass, Kclass):
    numClasses = len(samplesPerClass)
    numSamplesPerClass = len(samplesPerClass[0])
    clocks = [samplesPerClass[i][0] for i in range(numClasses)]
    idxs = [0] * numClasses
    mergedReqList = []
    while np.all(np.array(idxs) < numSamplesPerClass):
        nextEvent = min(clocks)
        source = clocks.index(nextEvent)
        mergedReqList.append(Kclass[source])
        idxs[source] += 1
        if idxs[source] < numSamplesPerClass:
            clocks[source] += samplesPerClass[source][idxs[source]]
    return mergedReqList
    

def batchProbMontecarlo(matrixPerBuffer, B, Kclass, Kprob, numSamples, W):
    DEBUG = False
    #lastIdx = len(matrixPerBuffer) - 1
    #listD0 = matrixPerBuffer[lastIdx][0]
    #listD1 = matrixPerBuffer[lastIdx][1]
    listD0 = [mat[0] for mat in matrixPerBuffer]
    listD1 = [mat[1] for mat in matrixPerBuffer]
    #listD0 = matrixPerBuffer[:][0]#[lastIdx][0]
    #print(listD0)

    
    numClasses = len(listD0)
    numSamples = int(numSamples/numClasses)
    samplesPerClass = []
    #print(Kclass)
    for i in range(len(Kclass)):
        D0 = np.matrix(listD0[i])
        D1 = np.matrix(listD1[i])
        #print("{}\t{}\t{}".format(i, D0, D1))
        #print(Kclass[i])
        samplesPerClass.append(butools.map.SamplesFromMAP(D0, D1, numSamples))
    seqRequests = mergeLists(samplesPerClass, Kclass)

    minVal = 0
    batchSize = B
    if len(seqRequests) % batchSize == 0:
        maxVal = len(seqRequests)
    else:
        while len(seqRequests) % batchSize != 0:
            seqRequests = seqRequests[0:-1]
        maxVal = len(seqRequests)
    batch_dict = {}
    for i in range(minVal,maxVal,batchSize):
        comp = tuple(sorted(seqRequests[i:i+batchSize]))
        if comp in batch_dict.keys():
            batch_dict[comp] = batch_dict[comp] + 1
        else:
            batch_dict[comp] = 1
            
    total = sum(batch_dict.values())
    for key in batch_dict.keys():
        batch_dict[key] = batch_dict[key] / total
        
    if DEBUG:
        print('Observed samples: '+str(len(seqRequests)))
        
    filename = getFilename(B, Kclass, Kprob, W)
    save_obj(batch_dict, 'batchProb/'+filename)
    if DEBUG:
        print('The dictionary has been saved to ' + filename + '.pkl')


def initState(D0, D1, B, T, pi=None):
    phases = len(D0)
    events = [-1] * phases
    g = [-1] * phases
    pInit = [0] * (phases * B)
    if phases > 1:
        if pi is None:
            pi = solveCTMC(D0, D1)
        l = [sum(D1[i]) for i in range(phases)]
#         l = [D1[i][i] for i in range(phases)]
#         w = [-(D0[i][i]+D1[i][i]) for i in range(phases)]
        for i in range(phases):
            events[i] = float(l[i]) * float(pi[i])
            g[i] = events[i]/min(B, float(l[i]) * float(T) + 1)
        gTot = np.sum(g)
        for i in range(phases):
            pInit[i] = g[i]/gTot
    else:
        pInit[0] = 1.0
    return pInit


# Generate the Infinitesimal Generator Q of a MAP, given D0, D1, and the batch size B
def generateQ(D0, D1, B):
    if len(D0) != len(D1) or len(D0[0]) != len(D1[0]):
        print ("ERROR: This is not a valid MAP.")
        exit(-1)

    phases = len(D0)
    Z = np.zeros((phases, phases))

    tmp = []
    for i in range(B):
        l = []
        for j in range(B):
            if i==B-1 and j==i:
                l.append(Z)
            elif j==i:
                l.append(D0)
            elif j==i+1:
                l.append(D1)
            else:
                l.append(Z)
        tmp.append(np.hstack(tuple(l)))

    Q = np.vstack(tuple(tmp))

    return Q


# Get the batch size distribution considering the state space
def batchProbState(D0, D1, B, T, pi=None):
    phases = len(D0)
    if pi is None:
        pi = solveCTMC(D0, D1)
    pInit = initState(D0, D1, B, T, pi=pi)
    Q = generateQ(D0, D1, B)

    p = [0] * B
    pTmp = np.dot(pInit, expm(Q * T))
    i = 0
    j = 0
    while j < phases * B:
        if i < B - 1:
            p[i] = sum(pTmp[j:j+phases])
        else:
            p[i] = 1 - sum(p)
        i += 1
        j += phases

    return p


# Get the request size distribution starting from the batch size distribution
def reqProbState(D0, D1, B, T, pi=None):
    if pi is None:
        pi = solveCTMC(D0, D1)
    p = batchProbState(D0, D1, B, T, pi=pi)
    qTmp = [-1] * B
    q = [-1] * B
    for i in range(B):
        qTmp[i] = (i+1) * p[i]
    sum_qTmp = np.sum(qTmp)
    for i in range(B):
        q[i] = qTmp[i]/sum_qTmp

    return q

def load_service_time_parameters():
    params = {}
    with open ('regression-parameters.json' , 'r') as fp:
        params = json.load(fp)
    params["params"]
    return params["params"]
    

def read_service_time():
    service_time = {}
    #with open("service_time.json", "r") as fp:
    with open("service_time_256.json", "r") as fp:
        service_time_json = json.load(fp)
        service_time = service_time_json["service_time"]
    return service_time
servTimePerReqSize_dict = read_service_time()
def batchServiceTimeMulticlass(B, K, M):
    '''servTimePerReqSize_dict = load_service_time_parameters()
    if B == 0:
        print('Max batch size (B) must be at least 1.')
        raise StopExecution
    if not M in range(128,3008+1,64):
        print(str(M)+' MB memory is not supported by AWS Lambda.')
        raise StopExecution
    reqSize = dict_size[K] #dict_size is a global variable
    coeffs = servTimePerReqSize_dict[str(reqSize)]
    return coeffs['q'] + coeffs['b'] * B + coeffs['m'] * M + coeffs['b2'] *\
         B**2 + coeffs['bm'] * B * M + coeffs['m2'] * M**2 + coeffs['b3'] *\
         B**3 + coeffs['b2m'] * B**2 * M + coeffs['bm2'] * B * M**2 + coeffs['m3'] * M**3'''
    if str(M) not in servTimePerReqSize_dict:
        M = M+256
    return servTimePerReqSize_dict[str(M)][K][B-1]/1000.0


def batchProb(matrixPerBuffer, B, Kclass, Kprob, numSamples, W):
    batchProbMontecarlo(matrixPerBuffer, B, Kclass, Kprob, numSamples, W)



# It generates all class probabilities
def generateClassProb(matrixPerBuffer, Bmin, Bmax, Kclass, Kprob, W=None, numSamples=1000000):
    if round(sum(Kprob), 5) != 1.0:
        print('Class probabilities must sum to 1.')
        raise StopExecution
    mythreads = list()
    for b in range(Bmin, Bmax+1):
        #print("This is batch size {}".format(b))
        #batchProb(matrixPerBuffer, b, Kclass, Kprob, numSamples, W)
        x = threading.Thread(target=batchProb, args=(matrixPerBuffer, b, Kclass, Kprob, numSamples, W))
        mythreads.append(x)
        x.start()
    for thread in mythreads:
        thread.join()

    #for b in range(Bmin, Bmax+1):
    #    batchProb(matrixPerBuffer, b, Kclass, Kprob, numSamples, W)


# Superpose two MAPs
def superposeMAPs(D0_1, D1_1, D0_2, D1_2):
    if len(D0_1) != len(D0_2):
        return False
    L = len(D0_1)
    D0 = np.kron(D0_1, np.identity(L)) + np.kron(np.identity(L), D0_2)
    D1 = np.kron(D1_1, np.identity(L)) + np.kron(np.identity(L), D1_2)
    return D0, D1


# Return the name of the file to save/store batch probabilities
def getFilename(B, Kclass, Kprob, W):
    filename = 'prob_W' + W + '_B' + str(B) + '_K' + str(Kclass).replace(" ", "") + '_('
    for i in range(len(Kprob)):
        if i != len(Kprob) - 1:
            filename += str(int(Kprob[i]*100)) + ','
        else:
            filename += str(int(Kprob[i]*100)) + ')'
    return filename


# Return class probabilities.
# To compute probabilities, we derive the rate lambda of each class from its MAP. Then, we normalize all rates.
# This function implement the same strategy used in KPCtool
def getClassProbability(listD0, listD1):
    rateList = []
    for D0, D1 in zip(listD0, listD1):
        infgen = D0 + D1
        mc = markovChain(infgen)
        mc.computePi('linear') #We can also use 'power', 'krylov', or 'eigen'
        p = mc.pi
        ones = np.array([1] * len(D1[0]))
        rate = np.dot(np.dot(p, D1), ones.transpose())
        rateList.append(rate)
    tot = sum(rateList)
    return [rate/tot for rate in rateList]
        

# Compute the probability of each batch composition for any batch size <= B
# Kprob: it is the vector containing the probability that the next request arriving to the system is of type k
def batchProbStateMulticlass(D0, D1, B, T, Kclass, Kprob, W, pi=None):
    if len(Kclass) != len(Kprob):
        print('Number of classes ('+str(len(Kclass))+') is different from the number of probabilities ('+str(len(Kprob))+')')
        raise StopExecution
        
    if pi is None:
        pi = solveCTMC(D0, D1)
        
    batchProb = batchProbState(D0, D1, B, T, pi=pi)
    stateProbMulti = []
    compositionMulti = []
    for i in range(len(batchProb)):
        tmpProb = []
        tmpComp = []
        filename = getFilename(i+1, Kclass, Kprob, W)
        classProbSize = load_obj('batchProb/'+filename)
        for q in classProbSize:
            tmpProb.append(batchProb[i] * classProbSize[q])
            tmpComp.append(q)
        stateProbMulti.append(tmpProb)
        compositionMulti.append(tmpComp)
#     print('test -> '+str(classProbSize))
    return (compositionMulti, stateProbMulti)


# This function returns the size of a request of class K in KB
def getReqSize(K):
#     ### Lambda ###
#     reqSize = [i*12 for i in range(1,10+1)]
    ### Twitter Trace ###
    reqSize = [11, 22.1, 25.7, 30.4, 33.5, 41.9, 43.9, 44.7, 51.9, 56,\
         59.4, 63.5, 67.5, 68.6, 76, 79,81.6, 87.9, 91.9, 96, 99.8, 103.5,\
         106.9, 110.8, 115.9, 118.7, 123.8, 127.5, 131.0, 134.5, 139.9, 143.5]
    return reqSize[K]


# Derive the request latency CDF for every request class
# addTimeout: should the timeout be added to all requests (True) or to none (False)?
def latencyCdfMulticlass(D0, D1, B, T, Kclass, Kprob, W, mem, pi=None, addTimeout=True, normalize=True):
    #print("Start of the function")
    if pi is None:
        pi = solveCTMC(D0, D1)
    (tmpComp, tmpProb) = batchProbStateMulticlass(D0, D1, B, T, Kclass, Kprob, W, pi=pi)
    classServiceTime = []
    #print(B)
    #print(len(tmpComp))
    #input("Before starting the loop stop")
    for k in Kclass:
        #print("Class Loop")
        tmp = []
        speedList = []
        probList = []
        for i in range(B): # batch size
            #print("Batch Loop")
            my_start_time = time.time()
            waitTime = waitingTime(D0, D1, i+1, B, T, pi=pi)
            for j in range(len(tmpComp[i])):
                if j == len(tmpComp[i]): # for testing
                    print("My class \t",k, "\t My Batch \t", i)
                    print(Kclass)
                    print(B)
                    print(len(tmpComp[i]))
                    input("I am stopping")
                comp = tmpComp[i][j]
                if k in comp:
                    prob = tmpProb[i][j]
                    maxReqClass = max(comp)
                    speed = batchServiceTimeMulticlass(i+1,maxReqClass,mem)
                    if addTimeout:
                        speed += waitTime
                    numReqTmp = comp.count(k) #Count how many request of class k are in the analyzed batch
                    probTot = numReqTmp * prob #This accounts for multiple request of the same class in a batch
                    if i == 0: #Set F_X(t)=0 for t < min(batchServiceTime)
                        speedList.append(speed)
                        probList.append(0.0)
                        #print("First if")
                    if probTot > 0: #Compute F_X(t)
                        speedList.append(speed)
                        probList.append(probTot)
                        #print("second if")
                    if i+2 <= B: #This sets the starting point of the next F_X(t) interval
                        speedList.append(batchServiceTimeMulticlass(i+2,maxReqClass,mem))
                        probList.append(0.0)
                        #print("Third if")
                    if i+1 == B: #Set F_X(t)=1 for t > max(batchServiceTime) + T
                        if_time = time.time()
                        speedList.append(batchServiceTimeMulticlass(i+1,maxReqClass,mem)+T)
                        probList.append(0.0)
                        #print("Fourth if \t{}".format(time.time()-if_time))
            #print("Loop time {}".format(time.time()-my_start_time))
        #print("Class loop stop \t{}\t{}".format( k, Kclass))
        #input("Class loop stop \t{}\t{}".format( k, Kclass)) #testing
        probSum = sum(probList)
        probList = [p / probSum for p in probList]
        for x in range(len(probList)):
            tmp.append((speedList[x], probList[x]))
        tmp.sort()
        classServiceTime.append(tmp)
        
    if normalize:
        classServiceTime = normalizeOnReqSize(classServiceTime, Kclass) # Latency CDF is normalized over the request size
    #print("End of the function")
    return classServiceTime


#This function compute the waiting time (that depends on the timeout).
#Currently, it assumes a uniform distribution and returns the a portion of the timeout for each request (T/bs)
#if the observed batch size is smaller than the maximum one. It returns a portion of the timeout (tau/bs)
#if the observed batch size is equal to the maximum one.
#bs: observed batch size
#B: maximum batch size
#T: timeout
def waitingTime(D0, D1, bs, B, T, pi=None):
    if B == 1:
        return 0
    elif bs < B:
        return T/bs
    else:
        if pi is None:
            pi = solveCTMC(D0, D1)
        rates = [sum(D1[i]) for i in range(len(D1))]
        avgRate = np.sum([pi[i] * rates[i] for i in range(len(pi))])
        return min(T, (bs-1)/avgRate)/bs
    

# def solveCTMC(D0, D1=None, decimals=2):
def solveCTMC(D0, D1=None, decimals=10):
    if D1 is None:
        D0 = roundMatrix(D0)
        mat = copy.deepcopy(D0)
    else:
        D0 = roundMatrix(D0, decimals)
        D1 = roundMatrix(D1, decimals)
        mat = D0 + D1
    numStates = len(mat)
    M = np.vstack((np.transpose(mat)[:-1], np.ones(numStates)))
    b = np.vstack((np.zeros((numStates - 1, 1)), [1]))
    return np.transpose(np.linalg.solve(M, b))[0]


# Derive the request cost CDF for every request class (cost per request)
def costCdfMulticlass(D0, D1, B, T, Kclass, Kprob, W, mem=3008, pi=None, normalize=True):
    if pi is None:
        pi = solveCTMC(D0, D1)
    (tmpComp, tmpProb) = batchProbStateMulticlass(D0, D1, B, T, Kclass, Kprob, W, pi=pi)
    classCost = []
    for k in Kclass:
        tmp = []
        costList = []
        probList = []
        for i in range(B): # batch size
            for j in range(len(tmpComp[i])):
                comp = tmpComp[i][j]
                if k in comp:
                    prob = tmpProb[i][j]
                    maxReqClass = max(comp)
                    servTime = ceil_decimal(batchServiceTimeMulticlass(i+1,maxReqClass,mem), 1) #Service time rounded as done by AWS
                    numReqTmp = comp.count(k)
                    probTot = numReqTmp * prob
                    memGB = mem / 1024
                    numReqInBatch = len(comp)
                    costPerReq = (servTime * memGB * 0.0000166667 + 0.0000002) / numReqInBatch
                    costList.append(costPerReq)
                    probList.append(probTot)
        probSum = sum(probList)
        probList = [p / probSum for p in probList]
        for x in range(len(probList)):
            tmp.append((costList[x], probList[x]))
        tmp.sort()
        classCost.append(tmp)
    if normalize:
        classCost = normalizeOnReqSize(classCost, Kclass) # Cost CDF is normalized over the request size
    return classCost


# This function returns the MAP process for a target buffer.
# listD0: list of all D0 matrices
# listD1: list of all D1 matrices
# numBuffers: number of buffers in the considered system
# targetBuffer: buffer whose MAP process must be returned (idx of the list, i.e., from 0 to numBuffers-1)
def superposeOnBuffers(listD0, listD1, numBuffers, targetBuffer):
    if targetBuffer >= numBuffers:
        print('Buffer not available. Choose another buffer to observe.')
        raise StopExecution
    if not powerOfTwo(len(listD0)):
        print('The number of source is not a power of two. The current superpose technique does not support the superposition of matrices with different size.')
        raise StopExecution
    numSources = len(listD0)
    splitIdx = int(numSources / numBuffers) # Number of request classes per buffer
    if splitIdx == 1:
        return listD0[targetBuffer], listD1[targetBuffer]
    finalD0 = []
    finalD1 = []
    for i in range(targetBuffer*2, splitIdx, 2):
        D0_tmp1 = listD0[i]
        D0_tmp2 = listD0[i+1]
        D1_tmp1 = listD1[i]
        D1_tmp2 = listD1[i+1]
        D0, D1 = superposeMAPs(D0_tmp1, D1_tmp1, D0_tmp2, D1_tmp2)
        finalD0.append(D0)
        finalD1.append(D1)
    i = 0
    while len(finalD0) > 1:
        if i < len(finalD0)-1:
            D0_tmp1 = finalD0[i]
            D0_tmp2 = finalD0[i+1]
            D1_tmp1 = finalD1[i]
            D1_tmp2 = finalD1[i+1]
            D0, D1 = superposeMAPs(D0_tmp1, D1_tmp1, D0_tmp2, D1_tmp2)
            finalD0[i] = D0
            finalD1[i] = D1
            del finalD0[i+1]
            del finalD1[i+1]
            i += 1
        else:
            i = 0
    return finalD0[0], finalD1[0]


# This function returned a scaled MAP process
# TODO: Check that this is working also for MAP. I am sure it is working for MMPP
def scaleMAP(D0_init, D1_init, scaleFact):
    if scaleFact == 1:
        return D0_init, D1_init
    D1 = D1_init * scaleFact
    D0 = copy.deepcopy(D0_init)
    for i in range(len(D0)):
        D0[i][i] = D0_init[i][i] + sum(D1_init[i]) - sum(D1[i])
    return D0, D1
def solveOptimizationProblem(D0, D1, Kclass, Kprob, quantile, slo, constraint, W, pi=None, addTimeout=True):
    if constraint != 'latency' and constraint != 'cost':
        print('You can optimize either __latency__ or __cost__.')
        exit(-1)
        
    if pi is None:
        pi = solveCTMC(D0, D1)

    maxTime = 30.0
    err = 0.01
    
    Mmin = 1024
    Mmax = 7000
    Mstep = 256
    Tmin = 0.005
    Tmax = 0.5
    Tstep = 0.005
    Bmin = 1
    Bmax = 25
    Bstep = 1
    
    memories = [x for x in range(Mmin, Mmax+Mstep, Mstep)]
    timeouts = [x for x in np.arange(Tmin, Tmax+Tstep, Tstep)]
    batches = [x for x in range(Bmin, Bmax+Bstep, Bstep)]
    matLatency = np.zeros((len(memories), len(timeouts), len(batches)))
    matCost = np.zeros((len(memories), len(timeouts), len(batches)))

    count = 0
    total = len(memories) * len(timeouts) * len(batches)

    start = time.time()

    i = 0
    while i < len(memories):
        j = 0
        while j < len(timeouts):
            k = 0
            while k < len(batches):
                latCdf = latencyCdfMulticlass(D0, D1, batches[k], timeouts[j], Kclass, Kprob, W, pi=pi, addTimeout=addTimeout)
                matLatency[i][j][k] = getQuantile(mergeClassValues(latCdf, Kprob), quantile)
                costCdf = costCdfMulticlass(D0, D1, batches[k], timeouts[j], Kclass, Kprob, W, mem=memories[i], pi=pi)
#                 matCost[i][j][k] = getQuantile(mergeClassValues(costCdf, Kprob), quantile)
                matCost[i][j][k] = getMeanValue(mergeClassValues(costCdf, Kprob))
                count += 1
                printProgressBar(count, total, prefix = 'Processing:', suffix = 'Complete', length = 50)
                k += 1
            j += 1
        i += 1
    solution = minimizeTarget(memories, timeouts, batches, matLatency, matCost, slo, constraint)

    end = time.time()
    print('Time to solve the problem: '+str(end - start)+' seconds.')

    return solution


# If the computation is too long, we can choose to stay with a (possibly) sub-optimal solution.
# When we see that the latency (cost) has not improved for a while, we skip other configurations and return the current minimum
# DO and D1: Matrices defining the MAP process
# quantile: Quantile used for the SLO
# slo: SLO value
# constraint: measure (latency or cost) on which the SLO is defined
# waitStep: number of iterations during which no improvements on latency and cost must be observed to directly skip to the next configuration
def solveOptimizationProblemSuboptimal(D0, D1, Kclass, Kprob, quantile, slo, constraint, W, waitStep=2, pi=None, addTimeout=True, checkAllMemory=False, checkAllTimeout=False, checkAllBatch=False):
    if constraint != 'latency' and constraint != 'cost':
        print('You can optimize either __latency__ or __cost__.')
        exit(-1)
        
    if pi is None:
        pi = solveCTMC(D0, D1)

    maxTime = 30.0
    err = 0.01
    
    Mmin = 1024
    Mmax = 3008
    Mstep = 128
    Tmin = 0.005
    Tmax = 0.5
    Tstep = 0.005
    Bmin = 1
    Bmax = 10
    Bstep = 1
    
    memories = [x for x in range(Mmin, Mmax+Mstep, Mstep)]
    timeouts = [x for x in np.arange(Tmin, Tmax+Tstep, Tstep)]
    batches = [x for x in range(Bmin, Bmax+Bstep, Bstep)]
    matLatency = np.ones((len(memories), len(timeouts), len(batches))) * np.inf #All elements of the amtrix are initialized to infinite
    matCost = np.ones((len(memories), len(timeouts), len(batches))) * np.inf #All elements of the amtrix are initialized to infinite

    count = 0
    total = len(memories) * len(timeouts) * len(batches)

    start = time.time()

    i = 0
    skipMemory = False
    noImprovementMemoryCounter = 0
    minLatMemory = np.inf
    minCostMemory = np.inf
    while i < len(memories):
        if not skipMemory or checkAllMemory:
            j = 0
            skipTimeout = False
            noImprovementTimeoutCounter = 0
            minLatTimeout = np.inf
            minCostTimeout = np.inf
            while j < len(timeouts):
                if not skipTimeout or checkAllTimeout:
                    k = 0
                    skipBatch = False
                    noImprovementBatchCounter = 0
                    minLatBatch = np.inf
                    minCostBatch = np.inf
                    while k < len(batches):
                        if not skipBatch or checkAllBatch:
                            latCdf = latencyCdfMulticlass(D0, D1, batches[k], timeouts[j], Kclass, Kprob, W, pi=pi, addTimeout=addTimeout)
                            currLat = getQuantile(mergeClassValues(latCdf, Kprob), quantile)
                            matLatency[i][j][k] = currLat
                            costCdf = costCdfMulticlass(D0, D1, batches[k], timeouts[j], Kclass, Kprob, W, mem=memories[i], pi=pi)
#                             currCost = getQuantile(mergeClassValues(costCdf, Kprob), quantile)
                            currCost = getMeanValue(mergeClassValues(costCdf, Kprob))
                            matCost[i][j][k] = currCost
                            if currLat > minLatBatch and currCost > minCostBatch:
                                noImprovementBatchCounter += 1
                            else:
                                noImprovementBatchCounter = 0
                                minLatBatch = currLat
                                minCostBatch = currCost
                            if noImprovementBatchCounter == waitStep:
                                skipBatch = True
#                         count += 1
#                         printProgressBar(count, total, prefix = 'Processing:', suffix = 'Complete', length = 50)
                        k += 1
                if minLatBatch > minLatTimeout and minCostBatch > minCostTimeout:
                    noImprovementTimeoutCounter += 1
                else:
                    noImprovementTimeoutCounter = 0
                    minLatTimeout = minLatBatch
                    minCostTimeout = minCostBatch
                if noImprovementTimeoutCounter == waitStep:
                    skipTimeout = True
                j += 1
        if minLatTimeout > minLatMemory and minCostTimeout > minCostMemory:
            noImprovementMemoryCounter += 1
        else:
            noImprovementMemoryCounter = 0
            minLatMemory = minLatTimeout
            minCostMemory = minCostTimeout
        if noImprovementMemoryCounter == waitStep:
            skipMemory = True
        i += 1
    solution = minimizeTarget(memories, timeouts, batches, matLatency, matCost, slo, constraint)

    end = time.time()
    print('Time to solve the problem: '+str(end - start)+' seconds.')

    return solution


# This function minimize the target (latency or cost) when a constraint is on the other parameter (cost or latency)
def minimizeTarget(memories, timeouts, batches, matLatency, matCost, slo, constraint, threshold=0):
    minTarget = -1.0
    countOkay = 0
    countOptimal = 0
    countTotal = 0
    minTargetList = []
    optMem = []
    optTimeout = []
    optBatch = []
    constList = []

    i = 0
    while i < len(memories):
        j = 0
        while j < len(timeouts):
            k = 0
            while k < len(batches):
                latency = matLatency[i][j][k]
                cost = matCost[i][j][k]
                countTotal += 1
                if constraint == 'latency':
                    if latency < slo:
                        countOkay += 1
                        minTargetList.append(cost)
                        optMem.append(memories[i])
                        optTimeout.append(timeouts[j])
                        optBatch.append(batches[k])
                        constList.append(latency)
                        if cost < minTarget or minTarget == -1.0:
                            minTarget = cost
                elif constraint == 'cost':
                    if cost < slo :
                        countOkay += 1
                        minTargetList.append(latency)
                        optMem.append(memories[i])
                        optTimeout.append(timeouts[j])
                        optBatch.append(batches[k])
                        constList.append(cost)
                        if latency < minTarget or minTarget == -1.0:
                            minTarget = latency
                k += 1
            j += 1
        i += 1

    memList = []
    timeoutList = []
    batchList = []
    targetList = []
    targetConstList = []
    if minTarget == -1.0:
        print('The problem cannot be solved.')
        exit(-1)
    else:
        for trg, const, b, t, m in zip(minTargetList, constList, optBatch, optTimeout, optMem):
            if trg >= minTarget * (1-threshold) and trg <= minTarget * (1+threshold):
                countOptimal += 1
                memList.append(m)
                timeoutList.append(t)
                batchList.append(b)
                targetList.append(trg)
                targetConstList.append(const)
                
    sol = (memList, timeoutList, batchList, targetConstList, minTarget, countOkay, countOptimal, countTotal, targetList)
    return minimizeConstraint(sol)


def minimizeConstraint(sol):
    minConstr = min(sol[3])
    idxMinConstr = sol[3].index(minConstr)
    memory = sol[0][idxMinConstr]
    timeout = sol[1][idxMinConstr]
    batch = sol[2][idxMinConstr]
    target = sol[8][idxMinConstr]
    return (memory, timeout, batch, minConstr, target, sol[5], sol[6], sol[7])


def collapse(valProbs):
    i = len(valProbs) - 1
    out = [[valProbs[i][0], valProbs[i][1], valProbs[i][2]]]
    i -= 1
    while i >= 0:
        if valProbs[i][0] != out[0][0]:
            out.insert(0, [valProbs[i][0], valProbs[i][1], valProbs[i][2]])
        i -= 1
    return out


def mergeClassValues(listVals, Kprobs):
    tmpVal = []
    tmpProb = []
    tot = 0.0
    for j in range(len(Kprobs)):
        val = listVals[j]
        prob = Kprobs[j]
        for i in range(len(val)):
            tmpVal.append(val[i][0])
            tmpProb.append(val[i][1]*prob)
    out = []
    for i in range(len(tmpVal)):
        out.append([tmpVal[i], tmpProb[i]])
    out.sort()
    partial = 0.0
    for i in range(len(out)):
        partial += out[i][1]
        out[i].append(partial)
    return out


def averageLambda(D0, D1):
    Q = []
    for r1, r2 in zip(D0, D1):
        row = []
        for e1, e2 in zip(r1, r2):
            row.append(e1+e2)
        Q.append(row)
    p = solveCTMC(Q)
    return np.sum(np.dot(p, D1))
            
            
def getQuantile(vals, quantile):
    for i in range(len(vals)):
        if vals[i][2] >= quantile:
            return vals[i][0]
        
        
def getMeanValue(vals):
    out = 0.0
    for i in range(len(vals)):
        out += vals[i][0] * vals[i][1]
    return out


def normalizeOnReqSize(cdfs, Kclass):
    out = []
    for i in range(len(Kclass)):
        out.append([(cdf[0] / getReqSize(Kclass[i]), cdf[1]) for cdf in cdfs[i]])
    return out


def chunks(lst, n):
    if n == 0 or len(lst) % n != 0:
        print('The list cannot be divided into '+str(n)+' equal parts.')
        raise StopExecution
    n = int(len(lst) / n)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
def leadingZeros(num, numDecimals=20):
    stringFormat = '{:.'+str(numDecimals)+'f}'
    num = stringFormat.format(num).replace('.', '')
    count = 0
    for i in range(len(num)):
        if num[i] != '0':
            return count
        else:
            count += 1
    return -1 # Return -1 if num == 0


def ceil_decimal(num, decimal):
    out = round(num, decimal)
    if out < num:
        out += 0.1
    return out


def powerOfTwo(n):
    if n <= 0:
        return False
    else:
        return n & (n - 1) == 0
    
    
def roundMatrix(mat, decimals=2):
    mat = [[round(mat[i][j], decimals) for j in range(len(mat[i]))] for i in range(len(mat))]
    return np.matrix(mat)
    
    
# Given a quantile (Y coordinate of a new point) and two points, this function determines the X coordinate of
# the new point after deriving m and q from the known points
# p1: last known point before the quantile
# p2: next known point after the quantile
# quantile: the desired quantile
def alignedPoint(p1, p2, quantile):
    m = (p2[2] - p1[2]) / (p2[0] - p1[0])
    q = p2[2] - m * p2[0]
    return (quantile - q) / m


def getPercentile(latencies, classProbs, quantile):
    classProbs = [prob / sum(classProbs) for prob in classProbs]
    totLatency = mergeClassValues(latencies, classProbs)
    for i in range(len(totLatency)-1):
        if quantile > totLatency[i][2] and quantile <= totLatency[i+1][2]:
            if totLatency[i][0] == totLatency[i+1][0] and totLatency[i][2] != totLatency[i+1][2]:
                return totLatency[i][0] #It's the same even if we return totLatency[i+1][0]
            else:
                return alignedPoint(totLatency[i], totLatency[i+1], quantile)
    return totLatency[0][0]


def loadArrProcessBuffer(filepath, W, numBuffs):
    listD0 = []
    listD1 = []
    for buff in range(numBuffs):
        fnameD0 = filepath + 'W' + str(W) + '_' + str(numBuffs) + '-' + str(buff) + '_D0.txt'
        fnameD1 = filepath + 'W' + str(W) + '_' + str(numBuffs) + '-' + str(buff) + '_D1.txt'
        D0_tmp = np.loadtxt(fnameD0, dtype='f', delimiter=',')
        D1_tmp = np.loadtxt(fnameD1, dtype='f', delimiter=',')
        listD0.append(D0_tmp)
        listD1.append(D1_tmp)
    return listD0, listD1

def read_arr_process(fname):
    arrival_matrix = np.loadtxt(fname, dtype='f', delimiter=',')
    return arrival_matrix

def loadArrProcessBuffer_new(filepath, W, numBuffs):
    listD0 = []
    listD1 = []
    for buff in range(numBuffs):
        #print(os.listdir('./{}/'.format(filepath)))#('./{}/buffer{}/MAPs/'.format(filepath,buff)))
        fnameD0 = './{}/buffer{}/MAPs/'.format(filepath,buff)  + 'W' + str(W) + '_' + str(1) + '-' + str(buff) + '_D0.txt'
        fnameD1 = './{}/buffer{}/MAPs/'.format(filepath,buff) + 'W' + str(W) + '_' + str(1) + '-' + str(buff) + '_D1.txt'
        D0_tmp = read_arr_process(fnameD0)#np.loadtxt(fnameD0, dtype='f', delimiter=',')
        D1_tmp = read_arr_process(fnameD1)#np.loadtxt(fnameD1, dtype='f', delimiter=',')
        listD0.append(D0_tmp)
        listD1.append(D1_tmp)
    return listD0, listD1


def printResults(sol, constraint, quantile, slo, Kcl, Kprob, normalize=True):
    print('###### Buffer serving class(es) '+str(Kcl)+' arriving with probabilities '+str(Kprob)+' ######')
    if len(sol) == 5:
        if constraint == 'latency':
            if normalize:
                print('SLO on '+str(quantile*100)+'th latency = '+str(slo)+' s/KB')
                print('Minimum Cost [USD/(r*KB)] = '+str(sol[4]))
            else:
                print('SLO on '+str(quantile*100)+'th latency = '+str(slo)+' s')
                print('Minimum Cost [USD/r] = '+str(sol[4]))
        else:
            if normalize:
                print('SLO on '+str(quantile*100)+'th cost = '+str(slo)+' USD/(r*KB)')
                print('Minimum Latency [s/KB] = '+str(sol[4]))
            else:
                print('SLO on '+str(quantile*100)+'th cost = '+str(slo)+' USD/r')
                print('Minimum Latency [s] = '+str(sol[4]))
        print('Time [s] = '+str(sol[3]))
        print('Memory [MB] = '+str(sol[0]))
        print('Timeout [s] = '+str(sol[1]))
        print('Batch [req] = '+str(sol[2]))  
    elif len(sol) == 7:
        if constraint == 'latency':
            if normalize:
                print('SLO on '+str(quantile*100)+'th latency = '+str(slo)+' s/KB')
            else:
                print('SLO on '+str(quantile*100)+'th latency = '+str(slo)+' s')
            print('Solutions SLO compliant: '+str(sol[5])+' ('+str(float(sol[5]*100/sol[7]))+'%)')
            print('Optimal solutions: '+str(sol[6])+' ('+str(float(sol[6]*100/sol[7]))+'%)')
            print('Memory [MB] = '+str(sol[0]))
            print('Timeout [s] = '+str(sol[1]))
            print('Batch [req] = '+str(sol[2]))
            if normalize:
                print('Norm. Latency [s/KB] (≤ '+str(slo)+' s/KB) = '+str(sol[3]))
                print('Norm. Cost [USD/(req*KB)] = '+str(sol[4]))
            else:
                print('Latency [s] (≤ '+str(slo)+' s) = '+str(sol[3]))
                print('Cost [USD/req] = '+str(sol[4]))
        else:
            if normalize:
                print('SLO on '+str(quantile*100)+'th cost = '+str(slo)+' USD/(r*KB)')
            else:
                print('SLO on '+str(quantile*100)+'th cost = '+str(slo)+' USD/r')
            print('Solutions SLO compliant: '+str(sol[5])+' ('+str(float(sol[5]*100/sol[7]))+'%)')
            print('Optimal solutions: '+str(sol[6])+' ('+str(float(sol[6]*100/sol[7]))+'%)')
            print('Memory [MB] = '+str(sol[0]))
            print('Timeout [s] = '+str(sol[1]))
            print('Batch [req] = '+str(sol[2]))
            if normalize:
                print('Norm. Latency [s/KB] = '+str(sol[4]))
                print('Norm. Cost [USD/(req*KB)] (≤ '+str(slo)+' USD/(req*KB)) = '+str(sol[3])) 
            else:
                print('Latency [s] = '+str(sol[4]))
                print('Cost [USD/req] (≤ '+str(slo)+' USD/req) = '+str(sol[3]))
    print('################################################')
    

# Plot Latency CDF for each request type
def plotLatencyCdf(filename, TotKcl, latencies, classProbs, quantile, slo, save):
    classProbs = [prob / sum(classProbs) for prob in classProbs]
    totLatency = mergeClassValues(latencies, classProbs)
    xAxisTot = [totLatency[i][0] for i in range(len(totLatency))]
    yAxisTot = [totLatency[i][2] for i in range(len(totLatency))]
    for reqClass in TotKcl:
        classIdx = TotKcl.index(reqClass)
        xAxis = [latencies[classIdx][i][0] for i in range(len(latencies[classIdx]))]
        yAxis = [latencies[classIdx][i][1] for i in range(len(latencies[classIdx]))]
        mean = np.dot(xAxis, yAxis)
        if len(xAxis) == 1:
            xAxis.insert(0, xAxis[0])
            yAxis.insert(0, 0)
        plt.plot(xAxis, np.cumsum(yAxis), label='Class '+str(TotKcl[classIdx])+' (Avg.={:.2E} s/KB)'.format(mean))#, where='post')
        with open('test_'+str(reqClass), 'w') as f:
            for x, y in zip(xAxis, np.cumsum(yAxis)):
                f.write(str(x)+'\t'+str(y)+'\n')
    plt.plot(xAxisTot, yAxisTot, '--', label='Global')#, where='post')
#     plt.hist(xAxisSim, bins=1000, density=True, cumulative=True, histtype='step', label='Simulation')
    plt.axvline(slo, ls='--', color='black')
    plt.axhline(quantile, ls='--', color='black')
    plt.xlabel('Latency [s/KB]')
    plt.ylabel('CDF')
    plt.ylim(0,1.1)
    quantileForTitle = getQuantile(mergeClassValues(latencies, classProbs), quantile)
    plt.title(str(int(quantile*100))+'th-percentile Latency = {:.2E} s/KB'.format(quantileForTitle))
    plt.grid()
    plt.legend()
    if save:
        plt.tight_layout()
        plt.savefig(filename+'_R.pdf')
    plt.show()
    return quantileForTitle

def new_fig(figsize=(8, 7), lblsz=25):
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(3)    
    ax1.yaxis.grid(linestyle='--')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(lblsz)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(lblsz)
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markersize(8)
    return ax1

def write_data(data_x, data_y, file_name):
    with open(file_name, 'w') as fp:
        for i in  range(min(len(data_x), len(data_y))):
            fp.write("{}\t{}\t{}\n".format(i,data_x[i], data_y[i]))

def calculate_cdf(data):
    count, bin_count = np.histogram(data, bins=1000)
    pdf = count /sum(count)
    cdf = np.cumsum(pdf)
    return bin_count, cdf
    

def plotLatencyCdfComparison(filename, TotKcl, latencies, classProbs, simResults, quantile, slo, save, addTimeout=True, classIndex=None):
    if classIndex is None:
        classProbs = [prob / sum(classProbs) for prob in classProbs]
    else:
        classProbs = [0.0] * len(classProbs)
        classProbs[classIndex] = 1.0
    totLatency = mergeClassValues(latencies, classProbs)
    xAxisTot = [totLatency[i][0] for i in range(len(totLatency))]
    yAxisTot = [totLatency[i][2] for i in range(len(totLatency))]
    xAxisSim = simResults
    yAxisSim = [i/len(simResults) for i in range(1,len(simResults)+1)]
    s_x, s_y = calculate_cdf(simResults)
    write_data(xAxisTot, yAxisTot, "{}_model".format(filename))
    write_data(s_x[1:], s_y, "{}_experiment".format(filename))
    return
    ax = new_fig(figsize=(5, 3))
    if not addTimeout:
        plt.plot(xAxisTot, yAxisTot, ':', label='No Timeout')#, where='post')
    else:
        plt.plot(xAxisTot, yAxisTot, ':', label='Model')#, where='post')
    #plt.hist(xAxisSim, bins=1000, density=True, cumulative=True, histtype='step', label='Experiment')
    s_x, s_y = calculate_cdf(simResults)
    plt.plot(s_x[1:], s_y, label='Experiment')
    plt.axvline(slo, ls='--', color='black')
    plt.axhline(quantile, ls='--', color='black')
    plt.xlabel('Latency [s/KB]', fontsize=22)
    plt.ylabel('CDF', fontsize=22)
    plt.ylim(0,1.1)
    quantileForTitle = getQuantile(totLatency, quantile)
    plt.title(str(int(quantile*100))+'th-percentile Latency = {:.2E} s/KB'.format(quantileForTitle))
    plt.grid()
    plt.legend()
    if save:
        plt.tight_layout()
        plt.savefig(filename+'_LatComp.pdf')
    plt.show()
    return quantileForTitle


def plotCostCdfComparison(filename, TotKcl, costs, classProbs, simResults, quantile, slo, save, classIndex=None):
    if classIndex is None:
        classProbs = [prob / sum(classProbs) for prob in classProbs]
    else:
        classProbs = [0.0] * len(classProbs)
        classProbs[classIndex] = 1.0
    totCost = mergeClassValues(costs, classProbs)
    xAxisTot = [totCost[i][0] for i in range(len(totCost))]
    yAxisTot = [totCost[i][2] for i in range(len(totCost))]
    xAxisSim = simResults
    yAxisSim = [i/len(simResults) for i in range(1,len(simResults)+1)]
    plt.plot(xAxisTot, yAxisTot, ':', label='Model')#, where='post')
    plt.hist(xAxisSim, bins=1000, density=True, cumulative=True, histtype='step', label='Simulation')
    plt.axvline(slo, ls='--', color='black')
    plt.axhline(quantile, ls='--', color='black')
    plt.xlabel('Cost [USD/(s*KB)]')
    plt.ylabel('CDF')
    plt.ylim(0,1.1)
    quantileForTitle = getQuantile(totCost, quantile)
#     plt.title(str(int(quantile*100))+'th-percentile Latency = {:.2E} s/KB'.format(quantileForTitle))
    plt.grid()
    plt.legend()
    if save:
        plt.tight_layout()
        plt.savefig(filename+'_CostComp.pdf')
    plt.show()
    return quantileForTitle


def plotCostCdf(filename, TotKcl, costs, classProbs, save):
    classProbs = [prob / sum(classProbs) for prob in classProbs]
    totCosts = mergeClassValues(costs, classProbs)
    xAxisTot = [totCosts[i][0] for i in range(len(totCosts))]
    yAxisTot = [totCosts[i][2] for i in range(len(totCosts))]
    for reqClass in TotKcl:
        classIdx = TotKcl.index(reqClass)
        xAxis = [costs[classIdx][i][0] for i in range(len(costs[classIdx]))]
        yAxis = [costs[classIdx][i][1] for i in range(len(costs[classIdx]))]
        mean = np.dot(xAxis, yAxis)
        if len(xAxis) == 1:
            xAxis.insert(0, xAxis[0])
            yAxis.insert(0, 0)
        plt.step(xAxis, np.cumsum(yAxis), label='Class '+str(TotKcl[classIdx])+' (Avg.={:.2E} $/(r*KB))'.format(mean), where='post')
    plt.step(xAxisTot, yAxisTot, '--', label='Global', where='post')
    plt.xticks(rotation=45)
    averageForTitle = getMeanValue(mergeClassValues(costs, classProbs))
    plt.title('Avg. Cost = {:.2E} $/(r*KB)'.format(averageForTitle))
    plt.xlabel('Cost [$/(r*KB)]')
    plt.ylabel('CDF')
    plt.ylim(0,1.1)
    plt.grid()
    plt.legend()
    if save:
        plt.tight_layout()
        plt.savefig(filename+'_C.pdf')
    plt.show()
    return averageForTitle


def plotComparison(filename, numBuffs, latQuantile, costAverage, quantile, save):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(numBuffs, latQuantile, '-o', color='blue', label=str(int(quantile*100))+'th-Perc. Latency')
    ax2.plot(numBuffs, costAverage, '-s', color='red', label='Avg. Cost')
    ax1Decimals = leadingZeros(max(latQuantile))
    ax1Max = (round(max(latQuantile), ax1Decimals)*(10**ax1Decimals)+1)/(10**ax1Decimals)
    ax1Step = ax1Max / (ax1Max * 10**ax1Decimals)
    if (ax1Max * 10**ax1Decimals) == 1:
        ax1Step = ax1Step / 2
    ax1Ticks = np.arange(0, ax1Max*1.1, ax1Step)
    ax1.set_yticks(ax1Ticks)
    ax2Decimals = leadingZeros(max(costAverage))
    ax2Max = (round(max(costAverage), ax2Decimals)*(10**ax2Decimals)+1)/(10**ax2Decimals)
    ax2Step = ax2Max / len(ax1Ticks)
    ax2Ticks = np.arange(0, ax2Max, ax2Step)
    ax2.set_yticks(ax2Ticks)
    xTicks = range(min(numBuffs), max(numBuffs)+1)
    ax1.set_xticks(xTicks)
    ax1.set_xlabel('Num. Buffers')
    ax1.set_ylabel('Latency [s/KB]')
    ax2.set_ylabel('Cost [$/(r*KB)]')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    if save:
        plt.tight_layout()
        plt.savefig(filename+'_Comp.pdf') 
    plt.show()
    
    
def plotQQ(filename, modelCdf, classProbs, simCdf, save):
    model = []
    simulation = []
    error = []
    for q in np.arange(0.01,0.99,0.01):
        newModel = getPercentile(modelCdf, classProbs, q)
        newSim = np.quantile(simCdf, q)
        newError = np.abs(newModel - newSim) / newSim
        model.append(newModel)
        simulation.append(newSim)
        error.append(newError)
    minVal = min(min(model), min(simulation))
    maxVal = max(max(model), max(simulation))
    plt.plot(model, simulation, 'o')
    plt.plot([minVal*0.9,maxVal*1.1], [minVal*0.9,maxVal*1.1], '--', color='black')
    plt.xlabel('Model')
    plt.ylabel('Simulation')
    plt.title('Avg. MAPE = {:.2f}%'.format(np.mean(error)*100))
    plt.grid()
    if save:
        plt.tight_layout()
        plt.savefig(filename+'_QQ.pdf')
    plt.show()
    
    
def plotErrorCdf(filename, modelCdf, classProbs, simCdf, save, classIndex=None):
    model = []
    simulation = []
    error = []
    quantiles = np.arange(0.01,0.99,0.01)
    '''if classIndex is None:
        totLatency = mergeClassValues(latencies, classProbs)
    else:
        totLatency = latencies[classIndex]'''
    for q in quantiles:
        newModel = getPercentile(modelCdf, classProbs, q)
        newSim = np.quantile(simCdf, q)
        newError = np.abs(newModel - newSim) * 100 / newSim
        model.append(newModel)
        simulation.append(newSim)
        error.append(newError)
    minVal = min(min(model), min(simulation))
    maxVal = max(max(model), max(simulation))
    s_x, s_y = calculate_cdf(error)
    write_data(s_x, s_y, "{}_model".format(filename))
    plt.hist(error, bins=len(quantiles), density=True, cumulative=True, histtype='step', lw=2)
    plt.xlabel('Error [%]')
    plt.ylabel('CDF')
    plt.yticks(np.arange(0,1.1,0.1))
    plt.title('Avg. MAPE = {:.2f}%'.format(np.mean(error)))
    plt.grid()
    if save:
        plt.tight_layout()
        plt.savefig(filename+'_ErrCdf.pdf')
    plt.show()



def get_configurations_actual(my_slo, folder_path, config_chuncks, my_probs):
    for W in ['7-8']:#, '1-2', '2-3', '3-4', '4-5', '5-6']:#, '6-7', '7-8', '8-9', '9-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-24']:
        DEBUG = False
        print("Configurations for SLO {}".format(my_slo))
        filepath = folder_path #'twitter_custom_length_32_maximum8_bimodal'
        #'./MAPs_length_4_maximum15_interval_0-6/' #_length_8_maximum12__mid2/'   
        # model = 'Resnet' # The model considered in the prediction (not used here)
        constraint = 'latency'
        quantile = 0.95
        slo = my_slo
        NEW_PROBS = True
        TIMEOUT = True
        numBuffsToConsider = [len(config_chuncks)]#[2**i for i in range(6)]
        TotKcl = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,\
            24,25,26,27,28,29,30,31] # All classes in the system (to change only when classes change)
        mixes = my_probs #[[1]*32] #[1,1,1,1], [36,1,1,2], [9,1,1,9], [2,1,1,36]
        my_chuncks = config_chuncks 
        
        mixesOrig = copy.deepcopy(mixes)
        #Normalize the mixes
        '''for i in range(len(mixes)):
            mix = mixes[i]
            sumMix = sum(mix)
            mix = [m/sumMix for m in mix]
            mixes[i] = mix'''
        #Store the matrices in listD0 and listD1
        matrixPerBuffer = []
        for buffs in numBuffsToConsider:
            #listD0, listD1 = loadArrProcessBuffer(filepath, W, buffs)
            listD0, listD1 = loadArrProcessBuffer_new(filepath, W, buffs)
            matrixPerBuffer.append((listD0, listD1)) #3D list: 1) buffer number; 2) D0 or D1; 3) request class served by the buffer
        matrixPerSource = []
        for indx, lst in enumerate(my_chuncks):
            sub_source =[]
            for r_indx,s in enumerate(lst):
                name = './{}/buffer{}/MAPs/'.format(filepath,indx)  + 'W' + str(W) + '_' + str(len(lst)) + '-' + str(s) + '_D0.txt'
                #print(name)
                listD0 = read_arr_process(name)
                name = './{}/buffer{}/MAPs/'.format(filepath,indx)  + 'W' + str(W) + '_' + str(len(lst)) + '-' + str(s) + '_D1.txt'
                listD1 = read_arr_process(name)
                #print(name)
                sub_source.append((listD0, listD1))
            #rint(sub_source)
            matrixPerSource.append(sub_source)
            #print(np.shape(matrixPerSource))
        #matrixPerSource = [loadArrProcessBuffer(filepath, W, len(TotKcl))]
        threads = list()
        if NEW_PROBS:
            #for mix in mixes:
            for mix in range(1):#mixes:
                #print(mix)
                for i, numBuffs in enumerate(numBuffsToConsider):
                    for j, Kcl in enumerate(my_chuncks):#chunks(TotKcl, numBuffs): # Classes per buffer
                        numCl = len(Kcl)
                        startIdx = TotKcl.index(Kcl[0])
                        endIdx = startIdx + numCl
                        #sumSubMix = sum(mix[startIdx:endIdx])
                        #Kprob = [round(p, 6)/sumSubMix for p in mix[startIdx:endIdx]]
                        sumSubMix = float(sum(my_probs[j]))
                        Kprob= [p/sumSubMix for p in my_probs[j]] 
                        #print(matrixPerSource[j])
                        generateClassProb(matrixPerSource[j], 1, 10, Kcl, Kprob, W, numSamples=100000)
                        #x = Process(target=generateClassProb, args=(matrixPerSource[j], 1, 25, Kcl, Kprob, W, numSamples=100000))
                        #threads.append(x)
                        #x.start()
            #for thread in threads:
            #    thread.join()

        mixConfigs = []
        #for mixID in range(len(mixes)):
        for mixID in range(1):
            numBuffsConfigs = []
            for i, numBuffs in enumerate(numBuffsToConsider):
                print("find config Buffer {}".format(numBuffs))
                config = []
                minTarget = []
                trgBuff = 0
                constraintParam = []
                classProbs = []
                for j, Kcl in enumerate(my_chuncks):#(chunks(TotKcl, numBuffs)): # Classes per buffer
                    #print(" Buffer id  {}".format(j))
                    numCl = len(Kcl)
                    startIdx = TotKcl.index(Kcl[0])
                    endIdx = startIdx + numCl
                    #sumSubMix = sum(mix[startIdx:endIdx])
                    #Kprob = [round(p, 6)/sumSubMix for p in mix[startIdx:endIdx]]
                    sumSubMix = float(sum(my_probs[j]))
                    Kprob= [p/sumSubMix for p in my_probs[j]] 
                    #print(Kcl)
                    #print()
                    D0 = matrixPerBuffer[i][0][j]
                    #print(D0)
                    D1 = matrixPerBuffer[i][1][j]
                    #print(D1)
                    pi = solveCTMC(D0, D1) #Get the state space for the CTMC of the derived MAP(2)

                    B = BayesOpt(D0, D1, pi, Kcl, Kprob, W, quantile, slo, constraint, 1, 26, 0.01, min(my_slo,0.5), 1024, 6912, 30, 30, numBuffs, j)

                    sol = BayesOpt(D0, D1, pi, Kcl, Kprob, W, quantile, slo, constraint, 1, 10, 0.005, 0.5, 1024, 3008, 50, 80).optimize()
                    config.append([sol[0], sol[1], sol[2]]) #Memory, Timeout, Batch
                    ###### START: print the constraint value ######
                    if constraint == 'latency':
                        constraintParamCDF = latencyCdfMulticlass(D0, D1, sol[2], sol[1], Kcl, Kprob, W, mem=sol[0], pi=pi, addTimeout=True)
                    elif constraint == 'cost':
                        constraintParamCDF = costCdfMulticlass(D0, D1, sol[2], sol[1], Kcl, Kprob, W, mem=sol[0], pi=pi, normalize=True)
                    for k in range(len(Kprob)): 
                        constraintParam.append(constraintParamCDF[k])
                        classProbs.append(Kprob[k])
                    ###### END: print the constraint value ######
                    trgBuff += 1
                    minTarget.append(sol[4])
                observedPercentile = getPercentile(constraintParam, classProbs, quantile)
                numBuffsConfigs.append([config, np.mean(minTarget), observedPercentile])
            mixConfigs.append(numBuffsConfigs)
            
        target = [mixConfigs[0][i][1] for i in range(len(mixConfigs[0]))]
        idxBuffer = target.index(min([mixConfigs[0][i][1] for i in range(len(mixConfigs[0]))]))
        print('######### Time Period: '+W+' #########')
        for i in range(len(numBuffsToConsider)):
            memoryConfig = []
            timeoutConfig = []
            batchConfig = []
            for config in mixConfigs[0][i][0]:
                memoryConfig.append(config[0])
                timeoutConfig.append(config[1])
                batchConfig.append(config[2])
            costConfig = mixConfigs[0][i][1]
            latencyConfig = mixConfigs[0][i][2]
            if i == idxBuffer:
                print('********* Number of buffers in the system: '+str(numBuffsToConsider[i])+' *********')
            else:
                print('--------- Number of buffers in the system: '+str(2**i)+' ---------')
            print('Cost = '+str(costConfig)+' USD/(req*KB)')
            print('Latency('+str(quantile*100)+'th) = '+str(latencyConfig)+' s/KB')
            print(memoryConfig)
            print(timeoutConfig)
            print(batchConfig)
            
        print('\n\n\n')



def get_configurations_parallel(my_slo, folder_path, config_chuncks, my_probs, new_prob):
    for W in ['7-8']:#, '1-2', '2-3', '3-4', '4-5', '5-6']:#, '6-7', '7-8', '8-9', '9-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-24']:
        DEBUG = False
        print("Configurations for SLO {}".format(my_slo))
        filepath = folder_path #'twitter_custom_length_32_maximum8_bimodal'
        #'./MAPs_length_4_maximum15_interval_0-6/' #_length_8_maximum12__mid2/'   
        # model = 'Resnet' # The model considered in the prediction (not used here)
        constraint = 'latency'
        quantile = 0.95
        slo = my_slo
        NEW_PROBS = new_prob #True 
        TIMEOUT = True
        numBuffsToConsider = [len(config_chuncks)]#[2**i for i in range(6)]
        TotKcl = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,\
            24,25,26,27,28,29,30,31] # All classes in the system (to change only when classes change)
        mixes = my_probs #[[1]*32] #[1,1,1,1], [36,1,1,2], [9,1,1,9], [2,1,1,36]
        my_chuncks = config_chuncks 
        
        mixesOrig = copy.deepcopy(mixes)
        #Normalize the mixes
        '''for i in range(len(mixes)):
            mix = mixes[i]
            sumMix = sum(mix)
            mix = [m/sumMix for m in mix]
            mixes[i] = mix'''
        #Store the matrices in listD0 and listD1
        matrixPerBuffer = []
        for buffs in numBuffsToConsider:
            #listD0, listD1 = loadArrProcessBuffer(filepath, W, buffs)
            listD0, listD1 = loadArrProcessBuffer_new(filepath, W, buffs)
            matrixPerBuffer.append((listD0, listD1)) #3D list: 1) buffer number; 2) D0 or D1; 3) request class served by the buffer
        matrixPerSource = []
        for indx, lst in enumerate(my_chuncks):
            sub_source =[]
            for r_indx,s in enumerate(lst):
                name = './{}/buffer{}/MAPs/'.format(filepath,indx)  + 'W' + str(W) + '_' + str(len(lst)) + '-' + str(s) + '_D0.txt'
                #print(name)
                listD0 = read_arr_process(name)
                name = './{}/buffer{}/MAPs/'.format(filepath,indx)  + 'W' + str(W) + '_' + str(len(lst)) + '-' + str(s) + '_D1.txt'
                listD1 = read_arr_process(name)
                #print(name)
                sub_source.append((listD0, listD1))
            #rint(sub_source)
            matrixPerSource.append(sub_source)
            #print(np.shape(matrixPerSource))
        #matrixPerSource = [loadArrProcessBuffer(filepath, W, len(TotKcl))]
        threads = list()
        if NEW_PROBS:
            #for mix in mixes:
            st = time.time()
            for mix in range(1):#mixes:
                #print(mix)
                for i, numBuffs in enumerate(numBuffsToConsider):
                    for j, Kcl in enumerate(my_chuncks):#chunks(TotKcl, numBuffs): # Classes per buffer
                        numCl = len(Kcl)
                        startIdx = TotKcl.index(Kcl[0])
                        endIdx = startIdx + numCl
                        #sumSubMix = sum(mix[startIdx:endIdx])
                        #Kprob = [round(p, 6)/sumSubMix for p in mix[startIdx:endIdx]]
                        sumSubMix = float(sum(my_probs[j]))
                        Kprob= [p/sumSubMix for p in my_probs[j]] 
                        #print(matrixPerSource[j])
                        #generateClassProb(matrixPerSource[j], 1, 10, Kcl, Kprob, W, numSamples=100000)
                        x = Process(target=generateClassProb, args=(matrixPerSource[j], 1, 25, Kcl, Kprob, W, 100000))
                        threads.append(x)
                        x.start()
            for thread in threads:
                thread.join()
            et = time.time()
            print("Time prob \t{}".format(et-st))

        mixConfigs = []
        #for mixID in range(len(mixes)):
        for mixID in range(1):
            numBuffsConfigs = []
            for i, numBuffs in enumerate(numBuffsToConsider):
                print("find config Buffer {}".format(numBuffs))
                config = []
                minTarget = []
                trgBuff = 0
                constraintParam = []
                classProbs = []
                my_threads = list()
                t_config = time.time()

                for j, Kcl in enumerate(my_chuncks):#(chunks(TotKcl, numBuffs)): # Classes per buffer
                    #print(" Buffer id  {}".format(j))
                    numCl = len(Kcl)
                    startIdx = TotKcl.index(Kcl[0])
                    endIdx = startIdx + numCl
                    #sumSubMix = sum(mix[startIdx:endIdx])
                    #Kprob = [round(p, 6)/sumSubMix for p in mix[startIdx:endIdx]]
                    sumSubMix = float(sum(my_probs[j]))
                    Kprob= [p/sumSubMix for p in my_probs[j]] 
                    #print(Kcl)
                    #print()
                    D0 = matrixPerBuffer[i][0][j]
                    #print(D0)
                    D1 = matrixPerBuffer[i][1][j]
                    #print(D1)
                    pi = solveCTMC(D0, D1) #Get the state space for the CTMC of the derived MAP(2)
                    #print("calculating for buffer\t{}".format(j))
                    B = BayesOpt(D0, D1, pi, Kcl, Kprob, W, quantile, slo, constraint, 1, 26, 0.01, min(my_slo,0.5), 1024, 6912, 30, 30, numBuffs, j)
                    #B.optimize()
                    x = Process(target= B.optimize, args=())
                    threads.append(x)
                    x.start()
                    #input("waiting for the first configurations")
                    #print("started")
                for thread in threads:
                    thread.join()
                st_config = time.time()
                print("Config time \t{}".format(st_config - t_config))

                config = [0]*numBuffs
                with open("temp_config", 'r') as fp:
                    for my_val in fp:
                        my_val = my_val.split()
                        config[int(my_val[1])] = [int(my_val[2]), float(my_val[3]), int(my_val[4]), float(my_val[5])] #Memory, Timeout, Batch, cost
                    
                
                for j, Kcl in enumerate(my_chuncks):#(chunks(TotKcl, numBuffs)): # Classes per buffer
                    numCl = len(Kcl)
                    startIdx = TotKcl.index(Kcl[0])
                    endIdx = startIdx + numCl
                    sumSubMix = float(sum(my_probs[j]))
                    
                    #Kprob = [p/float(sumSubMix) for p in mix[startIdx:endIdx]]
                    Kprob= [p/sumSubMix for p in my_probs[j]]
                    #print(len(Kprob))
                    D0 = matrixPerBuffer[i][0][j]
                    D1 = matrixPerBuffer[i][1][j]
                    pi = solveCTMC(D0, D1)
                    
                    if constraint == 'latency':
                        constraintParamCDF = latencyCdfMulticlass(D0, D1, config[j][2], config[j][1], Kcl, Kprob, W, mem=config[j][0]+512, pi=pi, addTimeout=True)
                    elif constraint == 'cost':
                        constraintParamCDF = costCdfMulticlass(D0, D1, config[j][2], config[j][1], Kcl, Kprob, W, mem=config[j][0]+512, pi=pi, normalize=True)

                    for k in range(len(Kprob)): 
                        constraintParam.append(constraintParamCDF[k])
                        classProbs.append(Kprob[k])

                    minTarget.append(config[j][3])
                observedPercentile = getPercentile(constraintParam, classProbs, quantile)
                #below three lines are for testing
                #sim_lat = read_data('./latency/reqLatency_buffexp-8-1.csv')
                #plotLatencyCdfComparison('test_latency', my_chuncks, constraintParam, classProbs, sim_lat, 0.95, 0.3, True, addTimeout=True, classIndex=None)
                #plotErrorCdf('test', constraintParam, classProbs, sim_lat, 'True', classIndex=None)
                my_mem =[]
                my_to = []
                my_bs = []
                for c in config:
                    my_mem.append(c[0])
                    my_to.append(c[1])
                    my_bs.append(c[2])
                print('--------- Number of buffers in the system: '+str(numBuffs)+' ---------') #str(2**i)
                print('Cost = '+str(np.mean(minTarget))+' USD/(req*KB)')
                print('Latency('+str(quantile*100)+'th) = '+str(observedPercentile)+' s/KB')
                print(my_mem)
                print(my_to)
                print(my_bs)
                os.remove("temp_config")
                
        return

def read_data(file_name):
    df = pd.DataFrame()
    df = pd.read_csv(file_name)
    sim_lat = df['latKB']
    return np.array(sim_lat)/1000.0

if __name__ == "__main__":
    slos = [0.3]#, 0.3, 0.5]#, 0.5]#[0.1, 0.3, 0.5]#,  0.065, 0.1, 0.2, 0.4,
    #chuncks = [[0, 1, 2, 3, 4, 5, 6], [7, 8], [8, 9, 10],\
    #  [10, 11, 12], [12, 13, 14], [14, 15, 16], [16, 17, 18,\
    #  19], [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]] #original
    #chuncks = [[0],[1,2],[3,4],[5,6,7, 8, 9, 10,11,12, 13, 14,15,16,17, 18,\
    #     19],[20,21, 22, 23, 24, 25,26], [26, 27,28], [28, 29,30], [30, 31]] #bimodal
    num_buffers = int(sys.argv[1])
    with open('requests_and_loads_random', 'r') as fp:
        chuncks_probs = json.load(fp)
    '''print(chuncks_probs["1"]['load'])
    print(chuncks_probs["1"]['requests'])
    chuncks = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [8, 9, 10, 11, 12], [12, 13, 14, 15, 16], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
    my_probs = [[0.00033620216488926383, 0.0007067966180448616, 0.0007706680105394816, 0.0035936431821148875, 0.015306673928820597, 0.035450728484816146,\
 0.057815542226009264, 0.06380471356838943, 0.07221503181637606], [0.0001351125610463133, 0.08204666401822652,\
  0.07305764287890126, 0.07471619343345937, 0.020044387108366535], [0.04861455228961398, 0.0699939216894615,\
   0.06609004602951563, 0.055731650310443026, 0.009569829680965869], [0.04184734316063732, 0.046055485289225095,\
    0.03540370229473766, 0.02980056684106136, 0.02382613505079534, 0.019556578121731838, 0.015910293682066423, 0.011349455127890207,\
    0.009099918721898304, 0.0071683355115116365,0.004507495413191731, 0.0031816376063529272, 0.0009047277464347969, 0.00036568126911751264,\
     0.0003776132874956373, 0.000645030875852215]]'''
    for i in [4, 8, 16, 20]:#range(num_buffers,21):#num_buffers+1):
        chuncks = chuncks_probs[str(i)]['requests']
        my_probs = chuncks_probs[str(i)]['load']
        new_probs= [True, False, False, False, False, False] #
        print(len(chuncks))
        #folder_path = 'MAPS_and_logs/orignial_routing/scale-4/twitter_custom_interval_7_8original_25'
        folder_path = 'MAPS_and_logs/percent_loading/random/Scale-1/buffers_{}'.format(i)
        for indx, slo in enumerate(slos):
            print("Configurations for SLO {}".format(slo))
            get_configurations_parallel(slo,folder_path, chuncks, my_probs,  new_probs[0])


        #for slo in slos:
        #    get_configurations_parallel(slo,folder_path, chuncks, my_probs)
        #for i in range(1,15):
        #    print("{}\t{}".format(i,batchServiceTimeMulticlass(i, 0, 3008)))