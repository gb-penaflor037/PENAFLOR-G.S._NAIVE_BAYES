import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/AB-105/Bayesian-Statistics/refs/heads/main/Naive%20Bayes%20Dataset.csv')
df
print(df)

df.shape
df.size
df.columns
df.info
df.describe
df.isna().sum()

sns.pairplot(df, hue = 'GRADE', plot_kws={'alpha':0.15})
plt.show()

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), cmap = 'YlGnBu', annot = True, square = True)
plt.show()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

y_test

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print("Classification Report: \n", classification_report(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print("Accuracy Score: ", acc*100, "%")