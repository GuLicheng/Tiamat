import time


def timer(func):
    """
        A decorate for calculating run time
    """
    def warp(*args):
        t1 = time.time()
        res = func(*args)
        t2 = time.time()
        delta = t2 - t1
        return res, f"{delta * 1000}ms"

    return warp

