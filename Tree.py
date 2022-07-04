from Node import *
import json
class BinaryDT:
    #> init method
    def __init__(self, num_nodes):
        self.nodes_list = [[]] * num_nodes
    
    #> Append a node to the tree
    def append_node(self, node_class):
        self.nodes_list[node_class.node_ID] = node_class
        
    #> Predict a single input
    def predict_single(self, X):
        current_node_id = 0
        current_node = self.nodes_list[current_node_id] 
        while not current_node.is_leaf:
            current_feature = current_node.feature
            current_threshold = current_node.threshold
            if X[current_feature] <= current_threshold:
                next_node_ID = current_node.l_child
            else:
                next_node_ID = current_node.r_child
            current_node = self.nodes_list[next_node_ID] 
        y_hat = current_node.get_class()
        return y_hat
    
    #> Predict for the whole data
    def predict(self, X):
        yhat = np.zeros(len(X), dtype=int)
        for idx in range(len(X)):
            yhat[idx] = self.predict_single(X[idx])
        return yhat
    
    #> Get the score of the tree
    def score(self, X, y):
        yhat = self.predict(X)
        currect_estimation = sum(1 for n in yhat - y if n == 0) 
        performance = currect_estimation / (len(y))
        return performance

    #> Store model as JSON
    def store_tree(self, store_path):
        #* convert all nodes to dictionary
        # tree_dict_list = [self.nodes_list[i].to_dict() for i in range(len(self.nodes_list))]
        tree_dict_list = []
        for i in range(len(self.nodes_list)):
            if isinstance(self.nodes_list[i], Node):
                # print(F"converting node {i}")
                tree_dict_list.append(self.nodes_list[i].to_dict())
        with open(store_path, 'w') as fout:
            json.dump(tree_dict_list , fout, indent = 4)

#> Read a json model and build the tree from it
def read_model(model_path):
    with open(model_path, 'r') as fin:
        in_dict_list = json.load(fin)
    num_nodes = len(in_dict_list)
    my_DT = BinaryDT(num_nodes)
    for in_dict in in_dict_list:
        current_node = from_dict(in_dict)
        my_DT.append_node(current_node)
    return my_DT

#> Building tree from a sklearn model
def from_sklearn(model):
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right

    num_nodes = model.tree_.node_count
    my_DT = BinaryDT(num_nodes)
    for node_idx in range(num_nodes):
        current_node = Node(node_idx, model.tree_.value[node_idx][0])
        #* check leaf status
        if children_left[node_idx] != children_right[node_idx]: # it is split
            current_node.make_inner_node(model.tree_.feature[node_idx], model.tree_.threshold[node_idx], children_left[node_idx], children_right[node_idx])
        else: # it is leaf 
            current_node.make_leaf_node()
        my_DT.append_node(current_node)
    return my_DT