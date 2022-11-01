from mpi4py import MPI
import numpy as np
import time
import sys
from seaborn import heatmap
import matplotlib.pyplot as plt

def getParts(rank, numberOfCores, gridLegth):

    lenthPerCore = 1.0 * gridLegth / numberOfCores
    indexes = (int(lenthPerCore* rank), gridLegth -1 if rank == numberOfCores else int(lenthPerCore * (rank+1)-1))
    return indexes

def communicate(comm, rank, numberOfCores, gridLegth, matrix, upPart, downPart, begin, end):
    if rank % 2 == 0:
        if rank > 0:
            comm.Send([matrix[0], gridLegth + 2, MPI.DOUBLE], dest=rank - 1)
        if rank < numberOfCores - 1:
            comm.Recv(upPart, source=rank + 1)
        if rank < numberOfCores - 1:
            comm.Send([matrix[end - begin], gridLegth + 2, MPI.DOUBLE], dest=rank + 1)
        if rank > 0:
            comm.Recv(downPart, source=rank - 1)
    else:
        if rank < numberOfCores - 1:
            comm.Recv(upPart, source=rank + 1)
        if rank > 0:
            comm.Send([matrix[0], gridLegth + 2, MPI.DOUBLE], dest=rank - 1)
        if rank > 0:
            comm.Recv(downPart, source=rank - 1)
        if rank < numberOfCores - 1:
            comm.Send([matrix[end - begin], gridLegth + 2, MPI.DOUBLE], dest=rank + 1)

def calculate(gridLegth, matrix, matrixTmp, upPart, downPart, begin, end):
    for y in range(begin, end + 1):
        for x in range(1, gridLegth + 1):
            if y - 1 == 0:
                newDown = 0
            elif y - 1 >= begin:
                newDown = matrix[y - 1 - begin][x]
            else:
                newDown = downPart[x]
            if y + 1 == gridLegth + 1:
                newUp = 0
            elif y + 1 <= end:
                newUp = matrix[y + 1 - begin][x]
            else:
                newUp = upPart[x]
            matrixTmp[y - begin][x] = (matrix[y - begin][x - 1] + matrix[y - begin][x + 1] + newUp + newDown) / 4
    return np.copy(matrixTmp)


def updateMatrix(comm, rank, numberOfCores, gridLegth, matrix, matrixTmp, upPart, downPart, begin, end):
    
    communicate(comm, rank, numberOfCores, gridLegth, matrix, upPart, downPart, begin, end)    
    matrix = calculate(gridLegth, matrix, matrixTmp, upPart, downPart, begin, end)
    return matrix


def getResult(comm, rank, numberOfCores, gridLegth, matrix, PartLenth, result):
    if rank != 0:
        comm.Send([matrix, PartLenth * (gridLegth + 2), MPI.DOUBLE], dest=0)
    else:
        index = 0

        for row in matrix:
            if row[1] == 0:
                break
            result[index] = row
            index += 1

        for i in range(numberOfCores - 1):
            temp = np.zeros((PartLenth, gridLegth + 2), dtype=np.float64)
            comm.Recv(temp, source=i + 1)

            for x in temp:
                if x[1] == 0:
                    break

                result[index] = x
                index += 1


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numberOfCores = comm.Get_size()

    gridLegth = int(sys.argv[1])
    conductorVoltage =  int(sys.argv[2])
    conductorSize =  int(sys.argv[3])
    iterations = 1000
    PartLenth = int(gridLegth / numberOfCores )+1 

    if rank == 0:
        times = 0

    for _ in range(numberOfTests):

        if rank == 0:
            start = time.time()

        begin, end = getParts(rank, numberOfCores, gridLegth)
        matrix = np.random.rand(PartLenth, gridLegth + 2)
        for y in range(end - begin + 1):
            matrix[y][0] = 0
            matrix[y][gridLegth + 1] = 0

        conductorSize = max(2, conductorSize)
        conductorMin, conductorMax =  gridLegth//2 - conductorSize//2, gridLegth//2 + conductorSize//2

        if begin <conductorMax and end > conductorMin:
            offset = max(0, conductorMin-begin)
            for y in range(min(conductorMax-begin, end-begin-1, conductorMax-conductorMin, end-conductorMin)):
                matrix[y+ offset][conductorMin+1:conductorMax+1] = conductorVoltage

        matrixTmp = np.zeros((PartLenth, gridLegth + 2), dtype=np.float64)
        upPart = np.zeros(gridLegth + 2, dtype=np.float64)
        downPart = np.zeros(gridLegth + 2, dtype=np.float64)
        result = np.zeros((gridLegth, gridLegth + 2), dtype=np.float64)

        for i in range(iterations):
            matrix = updateMatrix(comm, rank, numberOfCores, gridLegth, matrix, matrixTmp, upPart, downPart, begin, end)
            if begin <conductorMax and end > conductorMin:
                offset = max(0, conductorMin-begin)
                for y in range(min(conductorMax-begin, end-begin -1, conductorMax-conductorMin,end-conductorMin)):
                    matrix[y+ offset][conductorMin+1:conductorMax+1] = conductorVoltage

        getResult(comm, rank, numberOfCores, gridLegth, matrix, PartLenth, result)

        if rank == 0:
            times += time.time() - start
            # x = np.arange(1, gridLegth+1, 1)
            # y = np.arange(1, gridLegth+1,1) # transpose
            # x, y = np.meshgrid(x, y)
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')


            # ax.plot_surface(x, y, result[:,1:-1],cmap='viridis', edgecolor='none')
            # plt.show()

    if rank == 0:
        print("Took: ", times / numberOfTests)
        print(f"RESTULTS nr_of_cores {size} time {times/numberOfTests}")


if __name__ == '__main__':
    numberOfTests = 10
    main()