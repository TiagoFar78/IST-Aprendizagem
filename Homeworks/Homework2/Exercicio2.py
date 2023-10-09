import numpy as np
from scipy.io.arff import loadarff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = loadarff('column_diagnosis.arff')

df = pd.DataFrame(data[0])

le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

x = df.drop('class', axis=1)
y = df['class']

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# O KNeighborsClassifier ja usa a Euclidean distance e os pesos uniformes por default
knn1 = KNeighborsClassifier(n_neighbors=1)
knn5 = KNeighborsClassifier(n_neighbors=5)

cumulative_cm1 = np.zeros((3, 3))
cumulative_cm5 = np.zeros((3, 3))

for train_idx, test_idx in cv.split(x, y):
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    knn1.fit(x_train, y_train)
    knn5.fit(x_train, y_train)

    y_pred1 = knn1.predict(x_test)
    y_pred5 = knn5.predict(x_test)

    cm1 = confusion_matrix(y_test, y_pred1)
    cm5 = confusion_matrix(y_test, y_pred5)

    cumulative_cm1 += cm1
    cumulative_cm5 += cm5

diff_cm = cumulative_cm1 - cumulative_cm5

class_names = ["Hernia", "Spondylolisthesis", "Normal"]

plt.figure(figsize=(8, 6))
sns.heatmap(diff_cm, annot=True, fmt='g', cbar=True, xticklabels=class_names, yticklabels=class_names)
plt.title("Difference in Cumulative Confusion Matrices (k=1 - k=5)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("Exercicio2.png")
plt.show()