import numpy as np

# set params
theta = 1/100
pi = (1/2, (1/4)*theta, (1/4)*(1-theta), (1/4)*(1-theta), (1/4)*theta)
n = 50

# generate fake data
num_samples = 100
X = []
Y = []
for i in range(num_samples):
    counts = np.random.multinomial(n, pi) # generate (y1, y2, y3, y4, y5) ~ Mult(n, pi)
    Y.append(int(counts[1])) # set Y[i] = y2
    X.append((int(counts[0] + counts[1]), int(counts[2]), int(counts[3]), int(counts[4]))) # set X[i] = (y1 + y2, y3, y4, y5)

# given X, estimate theta using EM
theta_old = 1
num_iter = 100
for i in range(num_iter):

    # variable updates
    p_old = (theta_old/4) / ((1/2) + (theta_old/4))

    # compute theta_new = argmax_theta Q(theta, theta_old)
    num = 0
    denom = 0
    for data in X:
        num += data[3] + p_old * data[0]
        denom += data[1] + data[2] + data[3] + p_old * data[0]
    theta_new = num/denom

    # variable updates
    theta_old = theta_new

    print(f"Step {i+1}: {theta_old}")

print("-"*5)
print(f"Final theta: {theta_old}")