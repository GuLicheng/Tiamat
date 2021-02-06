from typing import *

# convert a int to list
# for 832 -> return [8, 3, 2] if base is 10


class number_convert:
    """
        This is just a simply convertion so it may not efficient
    """
    @staticmethod
    def int_to_list(x: int, base: int = 10) -> List[int]:
        assert isinstance(x, int) and isinstance(base, int)
        # for [0-9A-Z]
        res = []
        while x:
            res.append(x % base)
            x //= base
        res.reverse()
        return res

def test_for_above():
    print(number_convert.int_to_list(80))
    print(number_convert.int_to_list(80, 2))
    print(number_convert.int_to_list(80, 8))
    print(number_convert.int_to_list(11, 16))

if __name__ == '__main__':
    test_for_above()