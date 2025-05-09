

def bind_front(adaptor, function):
    def closure(*args):
        return adaptor(function, *args)
    return closure

def combine(sequence, *closures):
    for closure in reversed(closures):
        sequence = closure(sequence)
    return sequence

def pipeline(sequence, *closures):
    for closure in closures:
        sequence = closure(sequence)
    return sequence

if __name__ == "__main__":

    s = "1234"

    a = combine(s, list, bind_front(map, int))

    print(a)


