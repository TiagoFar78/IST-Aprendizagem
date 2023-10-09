# Este codigo foi usado para calcular P(Y1 = 0.38, Y2 = 0.52 | B)

import numpy as np
from scipy.stats import multivariate_normal

mu = np.array([0.5925, 0.3275])
cov_matrix = np.array([[0.02289, -0.0097583], [-0.0097583, 0.03149]])

point = np.array([0.38, 0.52])

mvn = multivariate_normal(mean=mu, cov=cov_matrix)

pdf_value = mvn.pdf(point)

print("PDF at point (0.38, 0.52):", pdf_value)