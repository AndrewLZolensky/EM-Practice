import numpy as np

# set hyperparams
num_iter = 100
num_samples = 10000
probs = [1/2, 1/4, 1/4]
mus = [-5, 0, 5]
sds = [1, 1.25, 2]

# generate data
Z = []
X = []
for i in range(num_samples):
    cat = int(np.random.multinomial(1, probs).argmax())
    sample = float(np.random.normal(mus[cat], sds[cat]))
    Z.append(cat)
    X.append(sample)

import matplotlib.pyplot as plt

plt.hist(X, bins=100)
plt.show()