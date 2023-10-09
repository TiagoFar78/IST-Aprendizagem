# Este codigo foi usado para calcular P(Y1 = 0.42, Y2 = 0.59 | B)

import numpy as np
from scipy.stats import multivariate_normal

mu = np.array([0.5925, 0.3275])
cov_matrix = np.array([[0.02289, -0.0097583], [-0.0097583, 0.03149]])

point = np.array([0.42, 0.59])

mvn = multivariate_normal(mean=mu, cov=cov_matrix)

pdf_value = mvn.pdf(point)

print("PDF at point (0.42, 0.59):", pdf_value)