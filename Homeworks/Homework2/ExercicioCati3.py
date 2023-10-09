# Este codigo foi usado para calcular P(Y1 = 0.42, Y2 = 0.59 | A)

import numpy as np
from scipy.stats import multivariate_normal

mu = np.array([0.24, 0.52])
cov_matrix = np.array([[0.0064, 0.0096], [0.0096, 0.0336]])

point = np.array([0.42, 0.59])

mvn = multivariate_normal(mean=mu, cov=cov_matrix)

pdf_value = mvn.pdf(point)

print("PDF at point (0.42, 0.59):", pdf_value)