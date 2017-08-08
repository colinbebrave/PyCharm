#first import math
from math import *
def polysum(n,s):
    area = 0.25 * n * s * s / tan (pi / n)
    perimeter = n * s
#here I combine the code together,and I write perimeter * perimeter in case that function square is not available
    return round(area + perimeter * perimeter,4)