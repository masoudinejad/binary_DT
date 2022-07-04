import Tree as dtr

from sklearn import tree

# from sklearn.datasets import load_iris 
# X, y = load_iris(return_X_y=True)

# from sklearn.datasets import load_breast_cancer 
# X, y = load_breast_cancer(return_X_y=True)

from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

clf = tree.DecisionTreeClassifier(max_depth = 3)
clf = clf.fit(X, y)
print(clf.score(X, y))
my_DT1 = dtr.from_sklearn(clf)
print(my_DT1.score(X, y))

my_DT1.store_tree('./mm.json')
my_DT2 = dtr.read_model('./mm.json')
print(my_DT2.score(X, y))
