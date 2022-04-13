import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def integrate(func, limits, n=100):
    '''
    Args:
        func: the function to be integrated
        limits: list of limits for each variable in the form [[x1, x2], ... , [z1, z2]]
        n: step count, default n=100
    
    Returns: integration using Monte-Carlo sampling, variance
    '''
    n_params = len(limits) # variable number of params
    xi = [np.random.uniform(low=limit[0], high=limit[1], size=(n,)) for limit in limits] # generates a list of lists, xi for each param
    widths = [limits[param][1] - limits[param][0] for param in range(n_params)] # (b - a) for each param

    avg_width = 1
    for width in widths:
        avg_width = avg_width * width # multiplies width of each param together

    f = func(*xi) # passes list of xis into func
    f_squared = func(*xi) ** 2 # f**2 for variance

    return avg_width * np.mean(f), (np.mean(f_squared) - np.mean(f) ** 2) / n 


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
    
    return np.array(step)
        

def dim9(ax, ay, az, ba, by, bz, ca, cy, cz):
    a = np.array([ax, ay, az])
    b = np.array([ba, by, bz])
    c = np.array([ca, cy, cz])

    return 1/abs(np.dot(a + b, c.T))


def metropolis(f, p, A, x_initial, delta, n=1000):
    """
    Metropolis Integration.
    Args:
        f: integrand function
        p: sampling function
        x_initial: initial random walk position
        delta: sampling interval [-delta, delta]
        n: number of steps

    Returns: definite integral solution
    """

    x = np.ones(n) * x_initial
    for i in range(n-1):
        d = np.random.uniform(low=-delta, high=delta)
        x_trial = x[i] + d
        w = p(x_trial) / p(x_initial)
        if w >= 1:
            x[i+1] = x_trial
        else:
            r = np.random.uniform(low=0, high=1)
            if r <= w:
                x[i+1] = x_trial

    start = int(n * 0.5)
    I = np.mean((f(x) / p(x))[start::])

    return I


def f5(x):
    return 2 * np.exp(-(x**2))


def f5_weight(x):
    A = 1 / (2 - 2 * np.exp(-10))
    return np.exp(-abs(x)) * A


def f6(x):
    return 1.5 * np.sin(x)


def f6_weight(x):
    A = 3 / (2 * np.pi)
    return A * 4 * x * (np.pi - x) / np.pi**2


# Results
# Task 2
i2a, i2astd = integrate(f1, [[0, 1]], 10000)
print('2a: {:.3f} +- {:.3f}'.format(i2a, i2astd))
i2b, i2bstd = integrate(f2, [[0, 1]], 10000)
print('2b: {:.5f} +- {:.5f}'.format(i2b, i2bstd))
i2c, i2cstd = integrate(f3, [[-2, 2]], 10000)
print('2c: {:.5f} +- {:.5f}'.format(i2c, i2cstd))
i2d, i2dstd = integrate(f4, [[0, 1], [0, 1]], 10000)
print('2d: {:.5f} +- {:.5f}'.format(i2d, i2dstd))

# Task 3
i3a, i3astd = integrate(n_sphere, [[-2, 2], [-2, 2], [-2, 2]], n=10000)
print('3-sphere: {:.5f} +- {:.5f}'.format(i3a, i3astd))
i3b, i3bstd = integrate(n_sphere, [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], n=10000)
print('3-sphere: {:.5f} +- {:.5f}'.format(i3b, i3bstd))

# Task 4
n = 9
limits = [[0, 1] for _ in range(9)]
i4, i4std = integrate(dim9, limits, n=10000)
print('4: {:.5f} +- {:.5f}'.format(i4, i4std))

# Task 5a
n=100000
A = 1 / (2 - 2 * np.exp(-10))

'''
deltas = np.linspace(0, 5, 10000)
good_delta = []
for delta in deltas:
    i5a, delta = metropolis(f5, f5_weight, A, 0, delta, n)
    if delta[0] > 49.5 and delta[0] < 50.5:
        good_delta.append(delta)

print(good_delta)
'''

i5a = metropolis(f5, f5_weight, A, 0, 2.99, n)
print('5a: {}, n={}'.format(i5a, n))
x = np.linspace(-10, 10, n)
func = 2 * np.exp(-(x**2)) 
'''
fig, ax = plt.subplots()
ax.plot(x, func, label='func')
ax.hist(samples, bins=len(samples))
ax.legend()
plt.show()
'''

# Task 5b
A = 3 / (2 * np.pi)

i5b = metropolis(f6, f6_weight, A, 1.5, np.pi, n)
print('5b: {}, n={}'.format(i5b, n))
x = np.linspace(0, np.pi, n)
func = 1.5 * np.sin(x)




'''
fig, ax = plt.subplots()
ax.plot(x, func, label='func')
ax.hist(samples, bins=len(samples))
ax.legend()
plt.show()
'''

# Task 6
ns = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
for n in ns:
    val = 3.544907701811032
    i6a, i6astd = integrate(f5, [[-10, 10]], n)
    print("6a: {} +- {}, n = {}".format(i6a, i6astd, n))

for n in ns:
    i6b, i6bstd = integrate(f6, [[0, np.pi]], n)
    print("6b: {} +- {}, n = {}".format(i6b, i6bstd, n))
