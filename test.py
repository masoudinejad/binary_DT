from Tree import BinaryDT
from Node import *

from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

clf = tree.DecisionTreeClassifier()
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


#current_data = 55
#yhat1 = clf.predict(X[current_data:current_data+1])
#print(yhat1)
#yhat2 = my_DT.predict(X[current_data:current_data+1])
#print(yhat2)

yhat1 = clf.predict(X)
yhat2 = my_DT.predict(X)
print(yhat1 - yhat2)