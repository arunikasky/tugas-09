from sklearn import tree

# Database: Gerbang Logika AND
# X = Data, y = Target

x = [[0 , 0, 0],
[0 , 5, 0],
[0 , 0, 5],
[0 , 5, 5],
[5 , 5, 0],
[5 , 0, 5],
[5 , 5, 5],
[10, 5, 5],
[5 , 10, 5],
[10, 10, 10]
]
y = [0,0,0,5,5,5,10,10,5,0]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

# Prediction
print("Logika AND Metode Decision Tree")
print("Logika = Prediksi")
print("10 10 5 = ", clf.predict([[10, 10, 5]]))
print("5 10 2 = ", clf.predict([[5, 10, 2]]))
print("2 0 10 = ", clf.predict([[2, 0, 10]]))
print("5 0 2 = ", clf.predict([[5, 0, 2]]))
print("0 0 2 = ", clf.predict([[0, 0, 2]]))
print("2 10 2 = ", clf.predict([[2, 10, 2]]))
print("1 12 5 = ", clf.predict([[1, 12, 5]]))
print("2 2 6 = ", clf.predict([[2, 2, 6]]))
print("10 5 7 = ", clf.predict([[10, 5, 7]]))
