def a():
    return 1, 2, 3

def b():
    x, _, _  = a()
    return x, _, _

print(b())
