from typing import *

class union_find:
    
    def __init__(self, n: int) -> None:
        self.parent = [-1 for _ in range(n)]
        self.num = n # connected component
        
    def merge(self, index1: int, index2: int) -> bool:
        fx, fy = self.find(index1), self.find(index2)
        if fx == fy and fx >= 0:
            return False # Not necessary to merge
        if fx > fy:
            fx, fy = fy, fx
        self.parent[fx] += self.parent[fy]
        self.parent[fy] = fx
        self.num -= 1
        return True

    def find(self, index: int) -> int:
        if self.parent[index] < 0:
            return index
        self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    # return connected components
    def count(self) -> int:
        return self.num

    def __repr__(self) -> str:
        return str(self.parent)

if __name__ == "__main__":
    uf = union_find(5)
    uf.merge(1, 2)
    uf.merge(1, 3)
    uf.merge(1, 4)
    print(uf)
    print(uf.count())