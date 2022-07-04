# from Tree import *
import Tree as tr
# from Node import *

from sklearn import tree

#from sklearn.datasets import load_iris 
#X, y = load_iris(return_X_y=True)

from sklearn.datasets import load_breast_cancer 
X, y = load_breast_cancer(return_X_y=True)

# from sklearn.datasets import load_digits
# X, y = load_digits(return_X_y=True)

clf = tree.DecisionTreeClassifier(max_depth = 5)
clf = clf.fit(X, y)

my_DT = tr.from_sklearn(clf)

p1 = clf.score(X, y)
print(p1)
p2 = my_DT.score(X, y)
print(p2)
