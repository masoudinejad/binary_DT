from Node import *
class BinaryDT:
    #> init method
    def __init__(self, num_nodes):
        nodes_list = [] * num_nodes
    
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
    
    