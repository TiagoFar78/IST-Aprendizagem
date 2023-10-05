from scipy.io.arff import loadarff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

data = loadarff('column_diagnosis.arff')

df = pd.DataFrame(data[0])

le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

x = df.drop('class', axis=1)
y = df['class']

clf = DecisionTreeClassifier(min_samples_leaf=20, random_state=0)
clf.fit(x, y)

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=x.columns, class_names=['Hernia', 'Spondylolisthesis', 'Normal'])
plt.title("Decision Tree Classifier")
plt.savefig("Exercicio4.png")
plt.show()