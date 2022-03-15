import numpy as np


def integrate(func, limits, n=100):
    '''
    func: the function to be integrated
    limits: list of limits for each variable in the form [[x1, x2], ... , [z1, z2]]
    n: step count, default n=100
    returns: integration using Monte-Carlo sampling
    '''
    n_params = len(limits) 
    xi = [np.random.uniform(low=limit[0], high=limit[1], size=(n,)) for limit in limits]
    widths = [limits[param][1] - limits[param][0] for param in range(n_params)]

    avg_width = 1
    for width in widths:
        avg_width = avg_width * width

    f = func(*xi)
    
    return avg_width * np.mean(f)


def f1(x):
    return 2     


def f2(x):
    return -x


def f3(x):
    return x**2


def f4(x, y):
    return x * y + x



def n_sphere(*dims):
    
    n_dims = len(dims)
    
    n_sum = 0
    for i in range(n_dims):
        n_sum += dims[i]**2 
    
    r = np.sqrt(n_sum) 
    
    step1 = [0 for x in r if x > 2]
    step2 = [1 for x in r if x <= 2]
    step = step1 + step2
    
    return step
        

def dim9(ax, ay, az, ba, by, bz, ca, cy, cz):
    a = np.array([ax, ay, az])
    b = np.array([ba, by, bz])
    c = np.array([ca, cy, cz])

    return 1/abs(np.dot(a + b, c.T))



i2a = integrate(f1, [[0, 1]], 10000)
print('2a: ', i2a)
i2b = integrate(f2, [[0, 1]], 10000)
print('2b: ', i2b)
i2c = integrate(f3, [[-2, 2]], 10000)
print('2c: ', i2c)
i2d = integrate(f4, [[0, 1], [0, 1]], 10000)
print('2d: ', i2d)
i3a = integrate(n_sphere, [[-2, 2], [-2, 2], [-2, 2]], n=10000)
print('3-sphere: ', i3a)
i3b = integrate(n_sphere, [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], n=10000)
print('5-sphere: ', i3b)

n = 9
limits = [[0, 1] for _ in range(9)]
i4 = integrate(dim9, limits, n=10000)
print('4: ', i4)


def f5(x):
    return 2 * np.exp(-(x**2))


def f6(x):
    return 1.5 * np.sin(x)


def f5_weight(x):
    return np.exp(-abs(x))


def f6_weight(x):
    return 


i5a = integrate(f5, [[-10, 10]], n=10000)
print('5a: ', i5a)
i6a = integrate(f6, [[0, np.pi]], n=10000)
print('6a: ', i6a)