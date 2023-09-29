from scipy.io.arff import loadarff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = loadarff('column_diagnosis.arff')

df = pd.DataFrame(data[0])

le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

x = df.drop('class', axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0, stratify=y)

depth_limits = [1, 2, 3, 4, 5, 6, 8, 10]

train_accuracies = []
test_accuracies = []

for depth in depth_limits:
    train_acc = 0
    test_acc = 0
    for _ in range(10):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(x_train, y_train)

        train_acc += accuracy_score(y_train, clf.predict(x_train))
        test_acc += accuracy_score(y_test, clf.predict(x_test))

    train_acc /= 10
    test_acc /= 10

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(depth_limits, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(depth_limits, test_accuracies, marker='o', label='Testing Accuracy')
plt.title('Decision Tree Accuracy vs. Depth Limit')
plt.xlabel('Depth Limit')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(depth_limits)

plt.savefig("Exercicio2.png")

plt.show()