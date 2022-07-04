import numpy as np
class Node:
    #> init method
    def __init__(self, node_ID, class_dist):
        #* Node ID is necessary
        self.node_ID = node_ID
        #+ Unknown leaf status
        self.is_leaf = None 
        #+ Children are unknown, set them to -1
        self.l_child = -1
        self.r_child = -1
        #+ Feature ID is not known
        self.feature = -1
        #+ No threshold is available yet
        self.threshold = float("nan")
        #* Distribution of classes is an essential element of a node (empty list possible but does not make sense) 
        self.class_dist = class_dist.astype(int)
        #+ Sample ID of the original data
        self.samples = []

    #> Setting the split data
    def set_split(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

    #> Setting children
    def set_children(self, l_child, r_child):
        self.l_child = l_child
        self.r_child = r_child

    # #> Setting the distribution of samples in each class
    # def set_class_dist(self, class_dist):
    #     self.class_dist = class_dist

    #> Set the list of samples ID which landed in this node
    def set_sample_list(self, samples):
        self.samples = samples

    #> Set the node as an inner-node
    def set_inner(self):
        self.is_leaf = False
    
    #> Set the node as a leaf
    def set_leaf(self):
        self.is_leaf = True

    #> Making an inner-node
    def make_inner_node(self, feature, threshold, l_child, r_child):#, samples):
        #* Set the split data
        self.set_split(feature, threshold)
        #* Set children data
        self.set_children(l_child, r_child)
        #! adding the samples list is optional (change it to that)
        #* Set the sample list
        #self.set_sample_list(samples)
        #* Set the status to inner node
        self.set_inner()

    #> Making a leaf
    def make_leaf_node(self): #, samples):
        #! adding the samples list is optional (change it to that)
        #* Set the sample list
        #self.set_sample_list(samples)
        #* Set the status to inner node
        self.set_leaf()
    
    #> Get majority class
    def get_class(self):
        y_hat = np.argmax(self.class_dist)
        return y_hat
    
    #> Making dict for storage as json
    def to_dict(self):
        node_dict = {"ID": int(self.node_ID),
                     "leaf": self.is_leaf,
                     "feat": int(self.feature),
                     "thre": float(self.threshold),
                     "left": int(self.l_child),
                     "right": int(self.r_child),
                     "dist": self.class_dist.tolist()
        }
        return node_dict
    
#> Making dict from a json
def from_dict(in_dict):
    current_node = Node(in_dict["ID"], np.array(in_dict["dist"]))
    current_node.is_leaf = in_dict["leaf"]
    current_node.feature = in_dict["feat"]
    current_node.threshold = in_dict["thre"]
    current_node.l_child = in_dict["left"]
    current_node.r_child = in_dict["right"]
    return current_node