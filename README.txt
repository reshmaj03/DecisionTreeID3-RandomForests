Following algorithms are implemented in this repository:
1. Naive Decision tree learner with Entropy as the impurity heuristic
2. Naive Decision tree learner with Variance as the impurity heuristic.
3. Decision tree learner with Entropy as the impurity heuristic and reduced error pruning
4. Decision tree learner with Variance as the impurity heuristic and reduced error pruning
5. Decision tree learner with Entropy as the impurity heuristic and depth-based pruning
6. Decision tree learner with Variance as the impurity heuristic and depth-based pruning
7. Random Forests

Variance impurity heuristic described below.
Let K denote the number of examples in the training set. Let K0 denote the number of training examples that have class = 0 and
K1 denote the number of training examples that have class = 1. The variance impurity of the training set S is defined as:
	VI(S) = (K0/K)(K1/K)


Datasets : https://www.hlt.utdallas.edu/~vgogate/ml/2020f/homeworks/hw1_data.zip

------------------------------------------------------------------------------------------------
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
