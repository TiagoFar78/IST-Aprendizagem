from scipy.io.arff import loadarff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = loadarff('column_diagnosis.arff')

df = pd.DataFrame(data[0])

le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

x = df.drop('class', axis=1)
y = df['class']

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
nb_classifier = GaussianNB()

knn_accuracies = []
nb_accuracies = []

for train_idx, test_idx in cv.split(x, y):
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    knn_classifier.fit(x_train, y_train)
    knn_pred = knn_classifier.predict(x_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_accuracies.append(knn_accuracy)

    nb_classifier.fit(x_train, y_train)
    nb_pred = nb_classifier.predict(x_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_accuracies.append(nb_accuracy)

results_df = pd.DataFrame({'Classifier': ['k-NN (k=5)'] * 10 + ['Naive Bayes (Gaussian)'] * 10, 'Accuracy': knn_accuracies + nb_accuracies})

plt.figure(figsize=(8, 5))
sns.boxplot(x='Classifier', y='Accuracy', data=results_df, width=0.2)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison between k-NN and Naive Bayes')
plt.savefig("Exercicio1.png")
plt.show()

_, p_value = stats.ttest_rel(knn_accuracies, nb_accuracies, alternative="greater")

print("is k-NN statistically superior to Naive Bayes regarding accuracy? (p-value =", p_value, ")")