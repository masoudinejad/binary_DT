from Tree import BinaryDT
from Node import *

from sklearn import tree

#from sklearn.datasets import load_iris 
#X, y = load_iris(return_X_y=True)

#from sklearn.datasets import load_breast_cancer 
#X, y = load_breast_cancer(return_X_y=True)

from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

clf = tree.DecisionTreeClassifier(max_depth = 4)
clf = clf.fit(X, y)

#print(clf.tree_.value[1])
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right

num_nodes = clf.tree_.node_count
my_DT = BinaryDT(num_nodes)
for node_idx in range(num_nodes):
    current_node = Node(node_idx, clf.tree_.value[node_idx])
    #* check leaf status
    if children_left[node_idx] != children_right[node_idx]: # it is split
        current_node.make_inner_node(clf.tree_.feature[node_idx], clf.tree_.threshold[node_idx], children_left[node_idx], children_right[node_idx])
    else: # it is leaf 
        current_node.make_leaf_node()
                
    my_DT.append_node(current_node)


#yhat1 = clf.predict(X)
#yhat2 = my_DT.predict(X)
#print(yhat1 - yhat2)

p1 = clf.score(X, y)
print(p1)
p2 = my_DT.score(X, y)
print(p2)
