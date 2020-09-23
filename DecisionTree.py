import numpy as np
from math import *

common_value = -1


class Node:
    def __init__(self, index=-1):
        self.left_child = None
        self.right_child = None
        self.index = index
        self.value = None
        self.number = None


def dtree_naive_entropy(train_data, test_data):
    instances, features = train_data.shape
    features -= 1
    attributes_to_decide = [1] * features
    root = dtree_entropy(train_data, attributes_to_decide)
    accuracy = calculate_accuracy(root, test_data)
    print("Accuracy obtained using Naive Decision Tree Learner with Entropy as impurity heuristic is", accuracy)


def dtree_naive_variance(train_data, test_data):
    instances, features = train_data.shape
    features -= 1
    attributes_to_decide = [1] * features
    root = dtree_variance(train_data, attributes_to_decide)
    accuracy = calculate_accuracy(root, test_data)
    print("Accuracy obtained using Naive Decision Tree Learner with Variance as impurity heuristic is", accuracy)


def dtree_entropy_rep(train_data, valid_data, test_data):
    instances, features = train_data.shape
    features -= 1
    attributes_to_decide = [1] * features
    root = dtree_entropy(train_data, attributes_to_decide)
    rep_root = dtree_rep(root, train_data, valid_data)
    accuracy = calculate_accuracy(rep_root, test_data)
    print(
        "Accuracy obtained using Decision Tree Learner with Entropy as impurity heuristic and Reduced Error Pruning is",
        accuracy)


def dtree_variance_rep(train_data, valid_data, test_data):
    instances, features = train_data.shape
    features -= 1
    attributes_to_decide = [1] * features
    root = dtree_variance(train_data, attributes_to_decide)
    rep_root = dtree_rep(root, train_data, valid_data)
    accuracy = calculate_accuracy(rep_root, test_data)
    print(
        "Accuracy obtained using Decision Tree Learner with Variance as impurity heuristic and Reduced Error Pruning is",
        accuracy)


def dtree_entropy_depth_prune(train_data, valid_data, test_data):
    instances, features = train_data.shape
    features -= 1
    attributes_to_decide = [1] * features
    root = dtree_entropy(train_data, attributes_to_decide)
    dp_root = dtree_depth_prune(root, train_data, valid_data)
    accuracy = calculate_accuracy(dp_root, test_data)
    print(
        "Accuracy obtained using Decision Tree Learner with Entropy as impurity heuristic and Depth-based Pruning is",
        accuracy)


def dtree_variance_depth_prune(train_data, valid_data, test_data):
    instances, features = train_data.shape
    features -= 1
    attributes_to_decide = [1] * features
    root = dtree_variance(train_data, attributes_to_decide)
    dp_root = dtree_depth_prune(root, train_data, valid_data)
    accuracy = calculate_accuracy(dp_root, test_data)
    print(
        "Accuracy obtained using Decision Tree Learner with Variance as impurity heuristic and Depth-based Pruning is",
        accuracy)


# --------- Naive Decision Tree Learner with Entropy as impurity heuristic --------
def dtree_entropy(data, attributes_to_decide):
    classes, classes_count = np.unique(data[:, -1], return_counts=True)
    instances, attribute_count = data.shape
    attribute_count -= 1

    if instances == 0:
        leaf_node = Node()
        leaf_node.value = common_value
        return leaf_node
    elif len(classes_count) == 1:
        leaf_node = Node()
        leaf_node.value = classes[0]
        return leaf_node
    elif attributes_to_decide.count(1) == 0:
        leaf_node = Node()
        max_count_index = classes_count.argmax()
        leaf_node.value = classes[max_count_index]
        return leaf_node
    else:
        best_attr = get_best_attribute_entropy(data, attribute_count)
        attributes_to_decide[best_attr] = 0
        attr_negative, attr_positive = split_data_on_attribute(data, best_attr)
        left_node = dtree_entropy(attr_negative, attributes_to_decide)
        right_node = dtree_entropy(attr_positive, attributes_to_decide)
        attributes_to_decide[best_attr] = 1
        internal_node = Node(best_attr)
        internal_node.left_child = left_node
        internal_node.right_child = right_node
        return internal_node


# --------- Naive Decision Tree Learner with Variance as impurity heuristic --------
def dtree_variance(data, attributes_to_decide):
    classes, classes_count = np.unique(data[:, -1], return_counts=True)
    instances, attribute_count = data.shape
    attribute_count -= 1

    if instances == 0:
        leaf_node = Node()
        leaf_node.value = common_value
        return leaf_node
    elif len(classes_count) == 1:
        leaf_node = Node()
        leaf_node.value = classes[0]
        return leaf_node
    elif attributes_to_decide.count(1) == 0:
        leaf_node = Node()
        max_count_index = classes_count.argmax()
        leaf_node.value = classes[max_count_index]
        return leaf_node
    else:
        best_attr = get_best_attribute_variance(data, attribute_count)
        attributes_to_decide[best_attr] = 0
        attr_negative, attr_positive = split_data_on_attribute(data, best_attr)
        left_node = dtree_variance(attr_negative, attributes_to_decide)
        right_node = dtree_variance(attr_positive, attributes_to_decide)
        attributes_to_decide[best_attr] = 1
        internal_node = Node(best_attr)
        internal_node.left_child = left_node
        internal_node.right_child = right_node
        return internal_node


# ------------- Reduced Error Pruning ------------
def dtree_rep(root, data, valid_data):
    best_tree = deep_clone(root)
    best_accuracy = round(calculate_accuracy(root, valid_data), 1)
    prev_best_accuracy = 0
    while prev_best_accuracy < best_accuracy:
        prev_best_accuracy = best_accuracy
        node_prune_list = assign_node_numbers(best_tree)
        prev_best_tree = deep_clone(best_tree)
        prune_nodes = len(node_prune_list)
        best_prune_node = -1
        # print("No of nodes", prune_nodes)
        # print("best_accuracy", best_accuracy)
        for i in reversed(range(prune_nodes)):
            pruned_tree = deep_clone(prev_best_tree)
            pruned_tree = prune_node(pruned_tree, i, data)
            current_accuracy = calculate_accuracy(pruned_tree, valid_data)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_prune_node = i
        best_accuracy = round(best_accuracy, 1)
        pruned_tree = deep_clone(prev_best_tree)
        best_tree = prune_node(pruned_tree, best_prune_node, data)
    # print("after prune, valid_acc", best_accuracy)
    return best_tree


# ------------- Depth Based Pruning ------------
def dtree_depth_prune(root, data, valid_data):
    d_max = [5, 10, 15, 20, 50, 100]
    original_tree = deep_clone(root)
    best_tree = deep_clone(root)
    max_depth = get_tree_depth(root)
    best_depth = d_max[0]
    best_accuracy = 0
    for i in d_max:
        if i > max_depth:
            # print(f"Maximum depth of tree is {max_depth}")
            break
        prune_tree = deep_clone(original_tree)
        prune_tree = prune_by_depth(prune_tree, i, data)
        accuracy = calculate_accuracy(prune_tree, valid_data)
        print(f"Accuracy on validation set with dmax {i} is {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_tree = deep_clone(prune_tree)
            best_depth = i
    print(f"Maximum accuracy on validation set is obtained for dmax {best_depth}")
    return best_tree


def get_common_value(data):
    class_data = data[:, -1]
    classes, classes_count = np.unique(class_data, return_counts=True)
    max_count_index = classes_count.argmax()
    return classes[max_count_index]


def get_best_attribute_entropy(data, attributes):
    highest_gain = -1
    best_attribute = -1
    entropy = calculate_entropy(data)
    for i in range(attributes):
        gain = calculate_information_gain(data, entropy, i)
        if gain > highest_gain:
            highest_gain = gain
            best_attribute = i
    return best_attribute


def calculate_entropy(data):
    class_data = data[:, -1]
    classes, classes_count = np.unique(class_data, return_counts=True)
    entropy = 0
    total_classes = classes_count.sum()
    for i in classes_count:
        pi = float(i / total_classes)
        entropy -= float(pi * log(pi, 2))
    return entropy


def calculate_information_gain(data, entropy, attribute):
    attr_negative, attr_positive = split_data_on_attribute(data, attribute)
    negative_len = len(attr_negative)
    positive_len = len(attr_positive)
    total_len = negative_len + positive_len
    p0 = float(negative_len / total_len)
    p1 = float(positive_len / total_len)
    attr_entropy = p0 * calculate_entropy(attr_negative) + p1 * calculate_entropy(attr_positive)
    return entropy - attr_entropy


def get_best_attribute_variance(data, attributes):
    highest_gain = -1
    best_attribute = -1
    variance_impurity = calculate_variance_impurity(data)
    for i in range(attributes):
        gain = calculate_information_gain_variance(data, variance_impurity, i)
        if gain > highest_gain:
            highest_gain = gain
            best_attribute = i
    return best_attribute


def calculate_variance_impurity(data):
    class_data = data[:, -1]
    classes, classes_count = np.unique(class_data, return_counts=True)
    if len(classes) == 1:
        return 0
    vi = 1
    k = classes_count.sum()
    for i in classes_count:
        vi *= float(i / k)
    return vi


def calculate_information_gain_variance(data, variance_impurity, attribute):
    attr_negative, attr_positive = split_data_on_attribute(data, attribute)
    negative_len = len(attr_negative)
    positive_len = len(attr_positive)
    total_len = negative_len + positive_len
    p0 = float(negative_len / total_len)
    p1 = float(positive_len / total_len)
    attr_variance_impurity = p0 * calculate_variance_impurity(attr_negative) + p1 * calculate_variance_impurity(
        attr_positive)
    return variance_impurity - attr_variance_impurity


def split_data_on_attribute(data, attribute):
    attribute_values = data[:, attribute]
    attr_negative = data[attribute_values == 0]
    attr_positive = data[attribute_values == 1]
    return attr_negative, attr_positive


def calculate_accuracy(root, data):
    instances, attr = data.shape
    attr -= 1
    correct_classifications = 0
    for i in range(instances):
        y = classify_instance(root, data[i])
        if y == data[i][-1]:
            correct_classifications += 1
    accuracy = float(correct_classifications / instances)
    return float(accuracy * 100)


def classify_instance(root, instance):
    if root.index == -1:
        return root.value
    if instance[root.index] == 0:
        return classify_instance(root.left_child, instance)
    else:
        return classify_instance(root.right_child, instance)


def assign_node_numbers(node):
    depth = get_tree_depth(node)
    node_prune_list = []
    for i in range(depth):
        set_node_number(node, i, node_prune_list)
    return node_prune_list


def get_tree_depth(node):
    if node.index == -1:
        return 0
    left_depth = get_tree_depth(node.left_child)
    right_depth = get_tree_depth(node.right_child)
    return (max(left_depth, right_depth)) + 1


def set_node_number(node, level, node_list):
    if node.index == -1:
        return
    if level == 0:
        node.number = len(node_list)
        node_list.append(1)
    else:
        set_node_number(node.left_child, level - 1, node_list)
        set_node_number(node.right_child, level - 1, node_list)


def deep_clone(node):
    if node is None:
        return None
    new_node = Node(node.index)
    new_node.left_child = deep_clone(node.left_child)
    new_node.right_child = deep_clone(node.right_child)
    new_node.value = node.value
    new_node.number = node.number
    return new_node


def prune_node(node, node_number, data):
    if node.index == -1:
        return node
    if node.number == node_number:
        instances, attribute_count = data.shape
        node.left_child = None
        node.right_child = None
        node.index = -1
        if instances == 0:
            node.value = common_value
        else:
            node.value = get_common_value(data)
        return node
    attr_negative, attr_positive = split_data_on_attribute(data, node.index)
    node.left_child = prune_node(node.left_child, node_number, attr_negative)
    node.right_child = prune_node(node.right_child, node_number, attr_positive)
    return node


def prune_by_depth(node, depth, data):
    if node.index == -1:
        return node
    if depth == 0:
        instances, attribute_count = data.shape
        node.left_child = None
        node.right_child = None
        node.index = -1
        if instances == 0:
            node.value = common_value
        else:
            node.value = get_common_value(data)
        return node
    else:
        attr_negative, attr_positive = split_data_on_attribute(data, node.index)
        node.left_child = prune_by_depth(node.left_child, depth - 1, attr_negative)
        node.right_child = prune_by_depth(node.right_child, depth - 1, attr_positive)
        return node
