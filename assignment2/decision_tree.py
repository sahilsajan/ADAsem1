#Assignment 2


import sys
import json
import math

#Nodes are in the form of a list of samples. The samples themselves are represented as a list of values

#Returns the entropy the current node
def entropy(samples):
    total=len(samples)
    label1={}
    for sample in samples:
        label= sample[-1]
        if label not in label1:
            label1[label]= 0
        label1[label] +=1
    my_entropy = 0
    for count in label1.values():
        p =count/total
        my_entropy-=p*math.log2(p) 
    return my_entropy

#Returns the Gini Index of the current node
def gini(samples):
    total= len(samples)
    label1={}
    for sample in samples:
        label = sample[-1]
        if label not in label1:
            label1[label]= 0
        label1[label]+=1
    my_gini= 1
    for count in label1.values():
        p= count /total
        my_gini -=p** 2
    return my_gini

#Calculate the weighted impurity of children of a candidate split, which are given as arguments
#num_0 is the number of samples in node 0 and num_1 is the number of samples in num_1
#impurity_0 and impurity_1 are the impurities of these nodes.
def impurity_of_children(impurity_0, impurity_1, num_0, num_1):
    total=num_0+num_1
    return (num_0/total)*impurity_0 +(num_1/total)*impurity_1
    #return my_impurity


#Return tuple of the most frequent label and the number of times it occurred
def get_most_frequent_label(samples):
    label1= {}
    for sample in samples:
        label= sample[-1]
        if label not in label1:
            label1[label] = 0
        label1[label] += 1
    most_freq_label= max(label1, key=label1.get)
    most_freq= label1[most_freq_label] 
    return (most_freq_label, most_freq)

#DecisionTreeNode class representd a node that will be added to a decision tree.
#Each node in your decision tree will be an instance of the class DecisionTreeNode
#When assigning label to empty node, assign the label of the parent
class DecisionTreeNode:
    def __init__(self, id='', samples=[], attr_remain=[], assigned_label=None):
        self.id = id #You don't need to use this, but it can help with picturing your decision tree
        self.leftChild = None #Assume that left child resulted in seeing a 0 for the valvue of the attribute
        self.rightChild = None #Assume that right child resulted in seeing a 1 for the valvue of the attribute
        self.samples = samples #A list of samples that are in this node
        self.is_leaf = True #Keeps track of whether the node is a leaf or not
        self.assigned_label = assigned_label #Is the assigned label to this node
        self.attribute_test = None #Sets to what attribute was used to split this node
        self.attributes_remaining = attr_remain #A list of attributes that have not been used for splitting
        self.combined_impurity_of_children = math.inf #The combined weighted impurity of children of this node

    #Based on a given impurity metric (gini or entropy), create the best binary split and create the two children nodes of the class DecisionTreeNodes.
    #You can use the gini and entropy functions that you create earlier
    #Update all attributes of the new children nodes
    def create_children(self, impurity_metric):
        best_split_attr= None
        best_impurity=math.inf
        best_left_split=None
        best_right_split= None
        for attr in self.attributes_remaining:
            left_split =[sample for sample in self.samples if sample[attr] == 0]
            right_split= [sample for sample in self.samples if sample[attr] == 1]
            if len(left_split) > 0 and len(right_split) > 0:
                if impurity_metric== 'gini':
                    left_impurity= gini(left_split)
                    right_impurity= gini(right_split)
                else:
                    left_impurity= entropy(left_split)
                    right_impurity= entropy(right_split)
                combined_impurity=impurity_of_children(left_impurity, right_impurity, len(left_split), len(right_split))
                if combined_impurity < best_impurity:
                    best_impurity= combined_impurity
                    best_split_attr =attr
                    best_left_split =left_split
                    best_right_split =right_split
        if best_split_attr is not None:
            self.is_leaf =False
            self.attribute_test=best_split_attr
            self.combined_impurity_of_children=best_impurity
            remaining_attrs=[attr for attr in self.attributes_remaining if attr != best_split_attr]
            if len(set([sample[-1] for sample in best_left_split]))==1:
                left_label=best_left_split[0][-1]
                self.leftChild=DecisionTreeNode(samples=best_left_split, attr_remain=remaining_attrs, assigned_label=left_label)
            else:
                most_freq_label_left=get_most_frequent_label(best_left_split)[0]
                self.leftChild=DecisionTreeNode(samples=best_left_split, attr_remain=remaining_attrs, assigned_label=most_freq_label_left)
            if len(set([sample[-1] for sample in best_right_split]))==1:
                right_label=best_right_split[0][-1]
                self.rightChild=DecisionTreeNode(samples=best_right_split, attr_remain=remaining_attrs, assigned_label=right_label)
            else:
                most_freq_label_right=get_most_frequent_label(best_right_split)[0]
                self.rightChild=DecisionTreeNode(samples=best_right_split, attr_remain=remaining_attrs, assigned_label=most_freq_label_right)


#Given a sample and a root node, predict a label
#Return the predicted label
def tree_classify(root_node, sample):
    current_node =root_node
    while not current_node.is_leaf:
        if sample[current_node.attribute_test]==0:
            current_node=current_node.leftChild
        else:
            current_node =current_node.rightChild
    return current_node.assigned_label

###TESTS
#DO NOT MODIFY ANY CODE BELOW THIS POINT

#build a tree
def buid_decision_tree_nodes(samples, attributes_remain, impurity_metric):

    full_dataset_assigned_label = get_most_frequent_label(samples)[0]
    root = DecisionTreeNode(id='-',samples = samples, attr_remain=attributes_remain, assigned_label=full_dataset_assigned_label)
    #Determine what to split on for the next level
    root.create_children(impurity_metric)

    root.leftChild.create_children(impurity_metric)

    root.rightChild.create_children(impurity_metric)

    return root

#Gets a list of attributes' (indices), not including the class label
def get_list_of_attributes(dataset):
    first_sample = dataset[0]
    attr_indices = list(range(1,len(first_sample)-1)) #ignores the first and last index, which are the name and the class
    return attr_indices

#Tests if two lists contain the same values
def same_two_lists(list_1, list_2):
    correct = True
    if(len(list_1) != len(list_2)):
        correct = False
    for curr in list_2:
        if(curr not in list_1):
            correct =  False
    
    return correct

if __name__ == '__main__':

    num_incorrect = 0
    num_tests = 6
    #Read in dataset
    dataset_file = sys.argv[1]
    dataset = json.load(open(dataset_file,'r'))
    #dataset is a list of samples. The samples themselves are represented as a list of values
    atribute_indices = get_list_of_attributes(dataset)


    root_entropy = buid_decision_tree_nodes(dataset, atribute_indices, 'entropy')

    #print(root_entropy.leftChild.rightChild.samples)


    left_right_child_samples = root_entropy.leftChild.rightChild.samples
    left_right_child_samples_correct = [['crab', 0, 0, 1, 0, 7], ['crayfish', 0, 0, 1, 0, 7], ['lobster', 0, 0, 1, 0, 7], 
                                        ['octopus', 0, 0, 1, 0, 7], ['seawasp', 0, 0, 1, 0, 7], ['starfish', 0, 0, 1, 0, 7]]

    correct = True
    if(not same_two_lists(left_right_child_samples, left_right_child_samples_correct)):
        print('Failed Check: grandchild node')
        num_incorrect += 1


    test_sample = ['sturgeon',0,0,1,1]
    prediction = tree_classify(root_entropy, test_sample)
    if(prediction != 4):
        print('Failed check: Tree Prediction 1')
        num_incorrect += 1

    test_sample_2 = ['housefly', 1, 0, 0, 0]
    prediction_2 = tree_classify(root_entropy, test_sample_2)
    if(prediction_2 != 6):
        print('Failed check: Tree Prediction 2')
        num_incorrect += 1

    if(round(root_entropy.combined_impurity_of_children, 2) != 0.55):
        print('Failed Check: combined_impurity_of_children')
        num_incorrect += 1

    few_samples = [['crab', 0, 0, 1, 0, 7], ['crayfish', 0, 0, 1, 0, 7], ['carp', 0, 0, 1, 1, 4]]
    #calculate the gini
    gini_samples = gini(few_samples)
   
    if(round(gini_samples,2) != 0.44):
        print('Failed Check: Gini')
        num_incorrect += 1

    #calculate the entropy
    entropy_samples = entropy(few_samples)
    if(round(entropy_samples,2) != 0.92):
        print('Failed Check: Entropy')
        num_incorrect += 1

    print(num_tests-num_incorrect,"out of",num_tests,"correctly passed")
    print('Grade:',(num_tests-num_incorrect)*2,'/',num_tests*2)
    
