import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("convergence.csv", delimiter=",")

iterations = data[:, 0]
values = data[:, 1]

plt.plot(iterations, values)

plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.title("Nelder-Mead Convergence")

plt.grid()
plt.show()