import collections
import sortedcontainers
'''
__all__ = ['deque', 'defaultdict', 'namedtuple', 'UserDict', 'UserList',
            'UserString', 'Counter', 'OrderedDict', 'ChainMap']
'''

deque_ = collections.deque()
deque_.append("right")
deque_.appendleft("left")

print(deque_)


