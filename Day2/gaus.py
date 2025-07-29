import numpy as np
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

mu = 0
sigma = 1
x = np.linspace(-5, 5, 100)
y = (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
plt.plot(x, y)
plt.title('Normal Distribution')
plt.xlabel('x')
plt.ylabel('PDF')
plt.grid(True)
plt.show()
