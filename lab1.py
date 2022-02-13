import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy.random import Generator, PCG64

# Lab 1 on random numbers, random walks, partitioned box


# Task 1 - uniformity and sequential correlation of random generators
pcg10 = np.random.default_rng(seed=10)
mt10 = np.random.Generator(np.random.MT19937(seed=10)) 

pcg50 = np.random.default_rng(seed=50)
mt50 = np.random.Generator(np.random.MT19937(seed=50)) 

pcg30 = np.random.default_rng(seed=30)
mt30 = np.random.Generator(np.random.MT19937(seed=30)) 

x10 = pcg10.random(10000)
y10 = mt10.random(10000)

x30 = pcg10.random(10000)
y30 = mt10.random(10000)

x50 = pcg10.random(10000)
y50 = mt10.random(10000)


def uniformity(x):
    y, m, b = plt.hist(x)
    N = len(x)
    M = len(m)
    E = N/M

    chaisq = 0
    for i in range(len(y)):
        chaisq += ((y[i] - E)**2) / E
    
    return chaisq - M


PCG64_uniformity = [uniformity(x10), uniformity(x30), uniformity(x50)]
MT19937_uniformity = [uniformity(y10), uniformity(y30), uniformity(y50)]
df = {'PCG64': [x10, x30, x50], 'MT19937': [y10, y30, y50]}
df = pd.DataFrame(df)

colors = ['red', 'green', 'blue']
fig, ax = plt.subplots(1, 2)
for i in range(3):
    sns.histplot(df['PCG64'][i], color=colors[i], bins=10, alpha=0.8, ax=ax[0], label='Uniformity {:.2f}'.format(PCG64_uniformity[i]))
    sns.histplot(df['MT19937'][i], color=colors[i], bins=10, alpha=0.8, ax=ax[1], label='Uniformity {:.2f}'.format(MT19937_uniformity[i]))

ax[0].set(title='PCG64')
ax[1].set(title='MT19937')
ax[0].legend()
ax[1].legend()
plt.show()

# Graphical
increment = np.array([1, 10])

for l in range(len(increment)):
    fig, ax = plt.subplots(1, 2, figsize=(4, 3))
    xni, xnf = [], []
    yni, ynf = [], []
    for i in range(len(x10) - increment[l]):
        xni.append(x10[i])
        xnf.append(x10[i + increment[l]])
        
        yni.append(y10[i])
        ynf.append(y10[i + increment[l]])

    ax[0].scatter(xni, xnf, c='k', s=1)
    ax[0].set(xlabel='$x_{n}$', ylabel='$x_{n+%s}$'%increment[l], title='PCG64 Sequential Correlation')

    ax[1].scatter(yni, ynf, c='k', s=1)
    ax[1].set(xlabel='$x_{n}$', ylabel='$x_{n+%s}$'%increment[l], title='MT19937 Sequential Correlation')

    plt.tight_layout()
    plt.show()

    
def partitioned_box(n=200, split=0.5, p=[0.5, 0.5]):
    '''
    Simulates the evolution of particles within a partitioned box
    n: no. of particles in the box, default 200
    split: defines the initial percentage of particles in box 1, default is 0.5
    p: the probability of moving a particle in a given box, default is p=[0.5, 0.5]
    '''

    N = np.linspace(0, n, n)
    t = np.linspace(0, 4000, 4000)

    # Initial conditions
    M2 = []
    M1 = N[0:int(len(N) * split)].tolist() # selects percentage of particles
    M2 = [x for x in N if x not in M1]
    
    N1 = [] # counters
    N2 = []
    partitions = [0, 1]
    for _ in t:
        partition = np.random.choice(partitions, p=p) # randomly selects partition with probability p
        particle = np.random.choice(N)
        if partition == 0:
            if particle in M1:
                M2.append(particle)
                M1.remove(particle)
                N2.append(len(M2))
                N1.append(len(M1))
        else: 
            if particle in M2:
                M1.append(particle)
                M2.remove(particle)
                N1.append(len(M1))
                N2.append(len(M2))

    time1 = np.linspace(0, 4000, len(np.array(N1)))
    time2 = np.linspace(0, 4000, len(np.array(N2)))
    midpoint = np.ones_like(time1) * len(N)/2
   
    return time1, time2, N1, N2, midpoint



# Task 2
no_particles = 500
results = partitioned_box(n=no_particles, split=1)

fig, ax = plt.subplots()
ax.set(ylabel='Number of particles in one side of container', xlabel='Time [arb.]')
ax.plot(results[0], results[4], 'k-', label='Partition wall')
ax.plot(results[0], results[2], c='c')
ax.set_ylim([0, no_particles*1.1])
plt.tight_layout()
ax.legend()
plt.show()

# Task 4
results = partitioned_box(n=no_particles, split=1, p=[0.75, 0.25])

fig, ax = plt.subplots()
ax.set(ylabel='Number of particles in one side of container', xlabel='Time [arb.]')
ax.plot(results[0], results[4], 'k-', label='Partition wall')
ax.plot(results[0], results[2], c='c')
ax.plot(results[1], results[3], c='y')
ax.set_ylim([0, no_particles*1.1])
plt.tight_layout()
ax.legend()
plt.show()
