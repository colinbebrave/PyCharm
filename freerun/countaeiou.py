s = 'azcbobobegghakl'

def co(x):
    count = 0
    for i in s:
        if i in 'aeiou':
            count += 1
    return count

print(co(s))

