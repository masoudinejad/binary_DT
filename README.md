# Binary Decision Tree
There is no standard for storing and sharing binary decision trees(DTs). 
Moreover, there is no established tool to convert a DT into the format from the sklearn.
This tool is to help building a general description of a binary DT which is framework independent.
Also, it can store/retrieve the DT into/from a JSON file.

Define structure is not including any training information (such as impurity in sklearn) which are not part of the structure.
However, it is possible to include reference to the ID of samples used in the training.
