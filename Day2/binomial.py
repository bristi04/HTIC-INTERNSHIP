import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n = 10      
p = 0.5     
x = np.arange(0, n+1) 

probabilities = binom.pmf(x, n, p)

plt.bar(x, probabilities, color='skyblue', edgecolor='black')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes (e.g., Heads)')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
