import math
import itertools
import more_itertools
import operator
import functools


def main():
    for i in range(1, 8):
        print(math.comb(2 * i, i) - math.comb(2 * i, i - 1), end="")
        print(" ", math.comb(2 * i, i) // (i + 1))


if __name__ == "__main__":
    main()
