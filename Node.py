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
        self.class_dist = class_dist
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
    def make_inner_node(self, feature, threshold, l_child, r_child, samples):
        #* Set the split data
        self.set_split(self, feature, threshold)
        #* Set children data
        self.set_children(l_child, r_child)
        #! adding the samples list is optional (change it to that)
        #* Set the sample list
        self.set_sample_list(samples)
        #* Set the status to inner node
        self.set_inner()

    #> Making a leaf
    def make_leaf_node(self, samples):
        #! adding the samples list is optional (change it to that)
        #* Set the sample list
        self.set_sample_list(samples)
        #* Set the status to inner node
        self.set_leaf()
    
    #> Get majority class
    def get_class(self):
        y_hat = np.argmax(self.class_dist)
        return y_hat