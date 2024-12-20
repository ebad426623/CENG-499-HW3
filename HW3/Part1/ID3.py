import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy_value = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """
        
        new_dataset = [d[attribute] for d in dataset]

        unique_attributes, unique_attributes_counts = np.unique([d[attribute] for d in dataset], return_counts=True)
        total_labels = len(labels)

        index = 0
        loop_size = len(unique_attributes)

        # Looping through each unique attribute
        while (index < loop_size):
            attribute_indices = [i for i in range(len(new_dataset)) if new_dataset[i] == unique_attributes[index]] 
            sub_labels = [labels[i] for i in attribute_indices]

            average_entropy += (unique_attributes_counts[index] / total_labels) * self.calculate_entropy__(sub_labels)
            index += 1

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """

        information_gain = self.calculate_entropy__(labels) - self.calculate_average_entropy__(dataset, labels, attribute)
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """

        total_data_instance = len(dataset)
        unique_attributes, unique_attributes_counts = np.unique([d[attribute] for d in dataset], return_counts=True)

        index = 0
        loop_size = len(unique_attributes)

        # Looping through each unique attribute
        while (index < loop_size):
            probablity = unique_attributes_counts[index] / total_data_instance
            if(probablity):
                intrinsic_info -= probablity * np.log2(probablity)
            index += 1

        return intrinsic_info
    

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """

        gain_ratio = 0
        information_gain = self.calculate_information_gain__(dataset, labels, attribute)
        intrinsic_info = self.calculate_intrinsic_information__(dataset, labels, attribute)

        if(intrinsic_info):
            gain_ratio = information_gain / intrinsic_info

        return gain_ratio


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """
        print("hi")

        if(len(np.unique(labels)) == 1):
            print("xd")
            return TreeLeafNode(dataset, labels[0])

        if len(used_attributes) == len(self.features):
            print("xD")
            unique_labels, unique_labels_counts = np.unique(labels, return_counts=True)
            index_of_max = np.argmax(unique_labels_counts)
            print(unique_labels, unique_labels_counts)
            return TreeLeafNode(dataset, unique_labels[index_of_max])
    
    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """
        
        # For debugging purpose
        d = self.dataset
        l = self.labels

        used_attributes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        
        
        node = self.ID3__(d, l, used_attributes)
        print(node.labels)
        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")