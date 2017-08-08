def f(x,y):
    return x > y
def dict_interdiff(d1,d2):
    interdict = {}
    diffdict = {}
    keylist = []
    for key1 in d1.keys():
        for key2 in d2.keys():
            if key1 == key2:
                keylist.append(key1)
                interdict[key1] = f(d1[key1],d2[key2])
    for key in d1.keys():
        if key not in keylist:
            diffdict[key] = d1[key]
    for key in d2.keys():
        if key not in keylist:
            diffdict[key] = d2[key]
    return (interdict,diffdict)
d1 = {1:30, 2:20, 3:30}
d2 = {1:40, 2:50, 3:60}
print dict_interdiff(d1,d2)