Use the below command to run all the algorithms:

	python DTreeAndRFClassifiers.py n <path to training set> <path to validation set> <path to test set>

where n can take values 1, 2, 3, 4, 5, 6 or 7
n is 1: Naive Decision tree learner with Entropy as the impurity heuristic
n is 2: Naive Decision tree learner with Variance as the impurity heuristic
n is 3: Decision tree learner with Entropy as the impurity heuristic and reduced error pruning
n is 4: Decision tree learner with Variance as the impurity heuristic and reduced error pruning
n is 5: Decision tree learner with Entropy as the impurity heuristic and depth-based pruning
n is 6: Decision tree learner with Variance as the impurity heuristic and depth-based pruning
n is 7: Random Forests


Example: 
To run Naive Decision tree learner with Variance as the impurity heuristic use the command below:

	python DTreeAndRFClassifiers.py 2 C:\Users\train_c1500_d100.csv C:\Users\valid_c1500_d100.csv C:\Users\test_c1500_d100.csv
