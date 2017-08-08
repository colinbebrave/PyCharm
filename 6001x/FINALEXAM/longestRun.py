def longestRun(L):
    count = 1
    maxcount = 1
    for i in range(len(L)-1):
        if L[i+1] >= L[i]:
            count += 1
        else:
            count = 1
        if maxcount < count:
            maxcount = count
    return maxcount

L = [10, 4, 6, 8, 3, 4, 5, 7, 7, 2]
print(longestRun(L))