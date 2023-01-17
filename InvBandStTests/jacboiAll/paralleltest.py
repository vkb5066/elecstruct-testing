from multiprocessing import Process, Queue


##Accumulate partial sums into an array
def PartialSum(ar, bounds, ind, q):
    su = q.get()
    for i in range(bounds[0], bounds[1]):
        su[ind] += ar[i]
    q.put(su)




if __name__ == '__main__':

    MIN = 0
    MAX = 100000000
    array = [i for i in range(MIN, MAX)]
    arrLen = len(array)

    #The no-doubt correct result
    print("ser ...")
    sum_ = 0
    for i in range(0, arrLen):
        sum_ += array[i]
    print("correct result:", sum_)

    #The threaded calculation
    print("par ...")
    N_THREADS = 8

    ##Sets an array of bounds for multithreads as [lo, hi] (lo inclusive, hi non-inclusive)
    bounds = [[None, None] for i in range(0, N_THREADS)]
    cur = 0
    dcur = arrLen//N_THREADS

    for i in range(0, N_THREADS):
        bounds[i][0] = cur
        cur += dcur
        bounds[i][1] = cur
    bounds[N_THREADS - 1][1] += arrLen - bounds[N_THREADS - 1][1]
    #print(bounds)


    accum = [0 for i in range(0, N_THREADS)]
    threads = [None for i in range(0, N_THREADS)]
    queue = Queue()
    queue.put(accum)
    for i in range(0, N_THREADS):
        threads[i] = Process(target=PartialSum, args=(array, bounds[i], i, queue))
        threads[i].start()
    for i in range(0, N_THREADS):
        threads[i].join()
    accum = queue.get()
    ##and then the final result is the sum of the partials

    fin = 0
    for i in range(0, N_THREADS):
        fin += accum[i]
    print("test result:", fin)
