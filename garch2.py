
import numpy as np
from numpy import exp, pi, sqrt, log, abs
from scipy.integrate import quad
from scipy.stats import norm
from scipy import randn
import matplotlib.pyplot as plt
from npkde import NPKDE
from scipy.integrate import quad, romberg

phi = norm.pdf

class GARCH:

    def __init__(self, a0=0.05, a1=0.05, b=0.9):
        self.a0, self.a1, self.b = a0, a1, b

    def q(self, y, x):
        return norm.pdf(y, scale=sqrt(x))

    def sim_bivariate(self, N):
        R = np.empty(N)
        X = np.empty(N)
        X[0] = 1
        for t in range(N-1):
            R[t] = sqrt(X[t]) * randn(1)
            X[t+1] = self.a0 + self.b * X[t] + self.a1 * R[t]**2
        R[N-1] = sqrt(X[N-1]) * randn(1)
        return R, X

    def psi(self, y, X_data):
        return np.mean(self.q(y, X_data))


## Main

# Number of replications and sample size 
num_reps = 50
sample_size = 500

# Plot range and plot grid
xmin, xmax = -4, 4
xgrid = np.linspace(xmin, xmax, 200)

# Create and instance of GARCH
gm = GARCH()

# Create an approximate stationary density
N = 10000
R_sim_long, X_sim_long = gm.sim_bivariate(N)
def approx_stat(y):
    return gm.psi(y, X_sim_long)

plt.subplot(121)
for m in range(num_reps):
    R_sim, X_sim = gm.sim_bivariate(sample_size)
    non_para = NPKDE(R_sim)
    plt.plot(xgrid, [non_para(y) for y in xgrid], color='0.6')
plt.plot(xgrid, [approx_stat(y) for y in xgrid], 'k-', linewidth=2, label=r"$\psi$")
plt.legend()
plt.ylim(0, 0.45)
plt.xlim(xmin, xmax)
plt.xticks([-3, 0, 3])
plt.yticks([0, 0.2, 0.4])
plt.title("NPKDE")

plt.subplot(122)
for m in range(num_reps):
    R_sim, X_sim = gm.sim_bivariate(sample_size)
    plt.plot(xgrid, [gm.psi(y, X_sim) for y in xgrid], color='0.6')
plt.plot(xgrid, [approx_stat(y) for y in xgrid], 'k-', linewidth=2, label=r"$\psi$")
plt.legend()
plt.ylim(0, 0.45)
plt.xlim(xmin, xmax)
plt.xticks([-3, 0, 3])
plt.yticks([0, 0.2, 0.4])
plt.title("GLAE")

plt.show()

