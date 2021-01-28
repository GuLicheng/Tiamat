import itertools
import functools
import operator

ls1 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
ls2 = list(ls1)

cmp = functools.cmp_to_key(lambda x, y: (x[0] < y[0]) or (x[0] == y[0] and x[1] > y[1]))
ls1.sort(key=lambda x: (x[0], -x[1]))
ls2.sort(key=cmp)
print(ls1)
print(ls2)
