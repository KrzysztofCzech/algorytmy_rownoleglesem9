#!/usr/bin/python
from mpi4py import MPI
import sys
import json
import math
import numpy as np

def erastotenes(upper_bound):
    array = np.ones([upper_bound+1, 1])
    i = 2
    while i * i <= upper_bound:
        j = i * i            
        while j <= upper_bound:
            array[j] = 0
            j += i
        i += 1
    primes = np.nonzero(array)[0][2:] 
    return  primes

def getCommBounds(comm_rank, comm_size, sqrt_upper_bound, upper_bound):
    NO_POINTS_PER_NODE = (upper_bound - sqrt_upper_bound) // comm_size
    min_bound = sqrt_upper_bound + comm_rank * NO_POINTS_PER_NODE
    if comm_rank != comm_size -1:
        return min_bound, min_bound + NO_POINTS_PER_NODE
    else:
        return min_bound, upper_bound

def find_primes(upper, lower, primes):
    array = np.ones([upper - lower +1, 1])
    for prime in primes:
        mod = lower % prime
        print(mod, lower, prime)
        idx =  (prime - mod)% prime
        while idx  <= upper - lower:
            array[idx] = 0
            idx += prime
    primes = np.nonzero(array)[0]
    return  primes + lower



comm = MPI.COMM_WORLD
upper_bound = int(sys.argv[1])
repeats = int(sys.argv[2])
sqrt_upper_bound = int(math.sqrt(upper_bound))
comm.barrier()
time1 = MPI.Wtime()

for i in range(0,repeats):
    prime_arr = erastotenes(sqrt_upper_bound)
    node_min, node_max = getCommBounds(comm.rank, comm.size, sqrt_upper_bound, upper_bound)
    node_primes = find_primes(upper= node_max, lower= node_min, primes = prime_arr)
    all_primes = comm.gather(node_primes,root=0)
time2 = MPI.Wtime()

if (comm.rank == 0):
    outputData = np.concatenate([prime_arr, *all_primes],axis = 0)
    print(outputData)
    data = { "primes": outputData, "time": (time2- time1)/repeats, "n_nodes": comm.size}
    json_object = json.dumps(data, indent = 2)
    print(json_object)


