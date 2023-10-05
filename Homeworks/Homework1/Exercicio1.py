from sklearn.feature_selection import f_classif
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = loadarff('column_diagnosis.arff')

df = pd.DataFrame(data[0])

x = df.drop('class', axis=1)
y = df['class']

f_values, p_values = f_classif(x, y)

highest_discriminative_idx = np.argmax(f_values)
lowest_discriminative_idx = np.argmin(f_values)

variable_names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]

most_discriminative_variable = variable_names[highest_discriminative_idx]
least_discriminative_variable = variable_names[lowest_discriminative_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.distplot(df[df["class"] == b'Spondylolisthesis'][most_discriminative_variable], hist=False, label="Spondylolisthesis", ax=ax1)
sns.distplot(df[df["class"] == b'Hernia'][most_discriminative_variable], hist=False, label="Hernia", ax=ax1)
sns.distplot(df[df["class"] == b'Normal'][most_discriminative_variable], hist=False, label="Normal", ax=ax1)
ax1.set_xlabel(most_discriminative_variable)
ax1.set_ylabel("Probability Density")
ax1.set_title(f"Class-Conditional PDF for {most_discriminative_variable} (Most discriminative)")

sns.distplot(df[df["class"] == b'Spondylolisthesis'][least_discriminative_variable], hist=False, label="Spondylolisthesis", ax=ax2)
sns.distplot(df[df["class"] == b'Hernia'][least_discriminative_variable], hist=False, label="Hernia", ax=ax2)
sns.distplot(df[df["class"] == b'Normal'][least_discriminative_variable], hist=False, label="Normal", ax=ax2)
ax2.set_xlabel(least_discriminative_variable)
ax2.set_ylabel("Probability Density")
ax2.set_title(f"Class-Conditional PDF for {least_discriminative_variable} (Least discriminative)")

ax1.legend()
ax2.legend()

plt.savefig("Exercicio1.png")
plt.tight_layout()
plt.show()