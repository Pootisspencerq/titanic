import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('Cleaned_Titanic.csv')


x = df.drop("Survived", axis=1)
y = df['Survived']


x_indices = x.index


x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(x, y, x_indices, test_size=0.25)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)


accuracy = accuracy_score(y_test, y_predict) * 100
print('Accuracy:', accuracy)


result = pd.DataFrame({"ID": idx_test, "Predicted": y_predict, "Actual": y_test.values})
result.to_csv("predict.csv", index=False)
