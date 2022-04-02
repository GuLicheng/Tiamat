from typing import Callable, Iterator, Iterable
import functools
import itertools

class CounterIterator(Iterator):

    def __init__(self, n: int) -> None:
        super().__init__()
        self.cur = 0
        self.sentinel = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur == self.sentinel:
            raise StopIteration()
        self.cur += 1
        return self.cur

class Sequence:
    """
        Simple linq that may not very efficient since python's iterator is just forward iterator
    """

    def __init__(self, sequence: Iterable) -> None:
        self.ls = sequence

    def for_each(self, fn: Callable):
        for val in self.ls: 
            fn(val)
        return self

    def where(self, fn: Callable):
        self.ls = filter(fn, self.ls)
        return self

    def count(self):
        self.exec()
        return len(self.ls)

    def to_list(self):
        return list(self.ls)

    def select(self, fn: Callable):
        self.ls = map(fn, self.ls)
        return self

    def exec(self):
        if not isinstance(self.ls, list):
            self.ls = list(self.ls)
    
    def take(self, n: int):
        self.ls = map(lambda x: x[1], zip(CounterIterator(n), self.ls))
        return self

    def skip(self, n: int):
        return self.take(self.count() - n)

    def take_while(self, fn: Callable):
        self.ls = itertools.takewhile(fn, self.ls)
        return self

    def skip_while(self, fn: Callable):
        self.ls = itertools.dropwhile(fn, self.ls)
        return self
        
    def reverse(self):
        self.ls = list(self.ls)
        self.ls.reverse()
        return self

    def zip(self, sequence: Iterable):
        self.ls = zip(self.ls, sequence)
        return self

    def distinct(self):
        self.exec()
        self.ls = list(set(self.ls)) 
        return self

    def concat(self, sequence: Iterable):
        self.ls = itertools.chain(self.ls, sequence)
        return self

    def first(self):
        self.exec()
        return self.ls[0]

    def last(self):
        self.exec()
        return self.ls[-1]



def make_sequence(sequence: Iterable):
    return Sequence(list(sequence))

if __name__ == "__main__":
    seq = make_sequence([1, 2, 3, 3, 3, 3, 3, 3]).distinct().concat([-1, -2]).for_each(print)


    for val in CounterIterator(4): print(val)
