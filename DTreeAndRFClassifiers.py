import sys
import pandas as pd
import numpy as np
import DecisionTree
import RandomForest


def print_help():
    print("Run as:\n\tpython DTreeAndRFClassifiers.py n <path to training set> <path to validation set> <path to test "
          "set>")
    print("where n can take values 1, 2, 3, 4, 5, 6 or 7")
    print("n is 1: Naive Decision tree learner with Entropy as the impurity heuristic")
    print("n is 2: Naive Decision tree learner with Variance as the impurity heuristic")
    print("n is 3: Decision tree learner with Entropy as the impurity heuristic and reduced error pruning")
    print("n is 4: Decision tree learner with Variance as the impurity heuristic and reduced error pruning")
    print("n is 5: Decision tree learner with Entropy as the impurity heuristic and depth-based pruning")
    print("n is 6: Decision tree learner with Variance as the impurity heuristic and depth-based pruning")
    print("n is 7: Random Forests")
    print("Example:\n\tpython DTreeAndRFClassifiers.py 2 C:\\Users\\train_c1500_d100.csv "
          "C:\\Users\\valid_c1500_d100.csv C:\\Users\\test_c1500_d100.csv")


if __name__ == '__main__':
    args = sys.argv

    if len(args) != 5:
        print_help()
    else:
        pd_train = pd.read_csv(args[2], header=None)
        pd_valid = pd.read_csv(args[3], header=None)
        pd_test = pd.read_csv(args[4], header=None)

        train_array = np.array(pd_train)
        valid_array = np.array(pd_valid)
        test_array = np.array(pd_test)

        DecisionTree.common_value = DecisionTree.get_common_value(train_array)

        if args[1] == "1":
            DecisionTree.dtree_naive_entropy(train_array, test_array)
        elif args[1] == "2":
            DecisionTree.dtree_naive_variance(train_array, test_array)
        elif args[1] == "3":
            DecisionTree.dtree_entropy_rep(train_array, valid_array, test_array)
        elif args[1] == "4":
            DecisionTree.dtree_variance_rep(train_array, valid_array, test_array)
        elif args[1] == "5":
            DecisionTree.dtree_entropy_depth_prune(train_array, valid_array, test_array)
        elif args[1] == "6":
            DecisionTree.dtree_variance_depth_prune(train_array, valid_array, test_array)
        elif args[1] == "7":
            RandomForest.classify_random_forest(pd_train, pd_test)
        else:
            print_help()
