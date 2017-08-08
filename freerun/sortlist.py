def swapSort(L):
    """ L is a list on integers
    sort L in increasing order
    """

    print "Original L: ", L
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if L[j] < L[i]:
                # the next line is a short
                # form for swap L[i] and L[j]
                L[j], L[i] = L[i], L[j]
                print L
    print "Final L: ", L

L = []

def modSwapSort(L):
    """ L is a list on integers
    sort L in decreasing order
    """
    print "Original L: ", L
    for i in range(len(L)):
        for j in range(len(L)):
            if L[j] < L[i]:
                # the next line is a short
                # form for swap L[i] and L[j]
                L[j], L[i] = L[i], L[j]
                print L
    print "Final L: ", L

modSwapSort(L)