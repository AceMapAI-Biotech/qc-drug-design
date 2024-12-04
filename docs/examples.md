---
layout: default
title: Usage Examples
---

# Basic Example: 2D Correlated Gaussian

This example demonstrates how to generate and visualize a two-dimensional correlated Gaussian distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

mean = [0, 0]
cov = [[1.0, 0.8],
       [0.8, 1.0]]

n_samples = 1000
samples = np.random.multivariate_normal(mean, cov, n_samples)

plt.figure(figsize=(10, 8))

plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, 
           label='Samples', color='blue', s=20)

plt.scatter(mean[0], mean[1], color='red', s=100, 
           label='Mean', marker='*')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Correlated Gaussian Distribution')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.text(0.05, 0.95, f'Correlation: 0.8', 
         transform=plt.gca().transAxes)

plt.savefig('2d_gaussian.png')
plt.close()
```

This code:
1. Generates samples from a 2D Gaussian with correlation coefficient 0.8
2. Creates a scatter plot of the samples
3. Marks the mean and adds relevant labels and grid