# from Tree import *
import Tree as tr
from Node import *

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

# tree_dict_list = [my_DT.nodes_list[i].to_dict() for i in range(len(my_DT.nodes_list))]
# for i in range(len(my_DT.nodes_list)):
#     print(i)
#     print(my_DT.nodes_list[i].to_dict())
# print(tree_dict_list)
my_DT.store_tree('./mm.json')

# p1 = clf.score(X, y)
# print(p1)
# p2 = my_DT.score(X, y)
# print(p2)

# print(clf.tree_.value[0][0])
# print(my_DT.nodes_list[0].class_dist)

# node_dict = my_DT.nodes_list[0].build_dict()
# print(node_dict)

import json
# node_json = json.dumps(node_dict, indent = 4)
# node_json_list =[my_DT.nodes_list[0].to_dict(), my_DT.nodes_list[1].to_dict(), my_DT.nodes_list[2].to_dict()]
# with open('./test.json', 'w') as fout:
#     json.dump(node_json_list , fout, indent = 4)
    
# with open('./test.json', 'r') as fin:
#     in_dict = json.load(fin)

# current_node = from_dict(in_dict[0])
# print(current_node)
# print(in_dict[0]["node_ID"])