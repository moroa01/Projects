import numpy as np
import pandas as pd
import math
import collections.abc

class Node:
    def __init__(self, decision_function=None, left: 'Node'=None, right: 'Node'=None, label=None, feature=None, threshold=None, impurity=None):

        if label is not None:
            self.is_leaf = True
        else:
            self.is_leaf = False

        if left is not None and not isinstance(left, Node):
            raise TypeError("'left' must be instance of 'Node' or None")
        
        if right is not None and not isinstance(right, Node):
            raise TypeError("'right' must be instance of 'Node' or None")

        self.feature = feature
        self.threshold = threshold
        self.decision_function = decision_function 
        self.left = left
        self.right = right
        self.label = label 
        self.impurity = impurity


    def predict(self, x):

        if self.is_leaf:
            return self.label
        if self.decision_function(x):
            return self.left.predict(x)
        else:
            return self.right.predict(x)



class TreeClassifier:

    def default_test_split(self, feature_value, split_value):

        if isinstance(feature_value, (int, float)): # If it's numerical
            if math.isnan(feature_value):           # If NaN is always false
                return False
            return float(feature_value) < float(split_value)
        elif isinstance(feature_value, str):        # If it's categorical
            return feature_value == split_value
        else:
            print("Type: ", type(feature_value), "\nValue: ", feature_value, "\n", type(split_value), split_value)
            raise ValueError("Unexpected feature type")

    def default_feature_selection(self, array):
        length = len(array)
        if(length <= 1):
            return array
    
        array = pd.Series(array)
    
        x_array = array.dropna().tolist()
    
        if(len(x_array) == 0):
            return array
    
        if isinstance(x_array[0], (int, float)):
            if(length > 4):
                new_array = np.sort(x_array)
                indices = np.linspace(0, len(new_array) - 1, 4, dtype=int)
                return new_array[indices]
    
        return np.sort(x_array)


    def __init__(self, max_depth=None, min_samples_split=2, splitting_criterion=None, feature_selection=None):

        if splitting_criterion is None:
            raise TypeError("'splitting_criterion' must be not None")

        if feature_selection is None:
            feature_selection = self.default_feature_selection

        self.max_depth = max_depth
        self.splitting_criterion = splitting_criterion
        self.feature_selection = feature_selection
        
        self.splitting_test = self.default_test_split
        self.min_samples_split = min_samples_split
        self.root = None


    def _stopping_criterion(self, depth, num_samples, y):
        
        if(self.max_depth is not None and self.max_depth > 0):
            if depth >= self.max_depth:
                return True

        if num_samples <= self.min_samples_split:
            return True

        if len(np.unique(y)) == 1:
            return True

        return False


    def fit(self, X, y):

        dataset = X.copy()
        dataset["predicted"] = y
        self.root = self._build_tree(dataset, depth=0)


    def _build_tree(self, dataset, depth):

        X, y = dataset.drop(columns=['predicted']), dataset["predicted"]
        num_samples, num_features = X.shape

        # Stopping condition so create a leaf
        if self._stopping_criterion(depth, num_samples, y):
            vect = np.array(y, dtype=int)
            leaf_value = np.bincount(vect).argmax()
            return Node(label=leaf_value)

        # Find optimal split
        feature_index, threshold, impurity = self._best_split(X, y)

        # Split the datas
        left_idxs, right_idxs = self._split(X[feature_index], threshold)
        
        left_child = self._build_tree(dataset.loc[left_idxs], depth + 1)
        right_child = self._build_tree(dataset.loc[right_idxs], depth + 1)

        # Decision Node
        decision_function = lambda x: self.splitting_test(x[feature_index], threshold)
        return Node(decision_function=decision_function, left=left_child, right=right_child, feature=feature_index, threshold=threshold, impurity=impurity)


    def _best_split(self, X, y):

        best_impurity = float('inf')
        best_index, best_threshold = None, None
        for i in X.columns.values:  # Loop on all the features
            thresholds = np.unique( self.feature_selection(X[i]) )
            if(len(thresholds)<1):
                continue

            for threshold in thresholds:
                if ( isinstance(threshold, float) and math.isnan(threshold) ):
                    continue
                impurity = self.splitting_function(X[i], y, threshold)
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_index = i
                    best_threshold = threshold
        return best_index, best_threshold, best_impurity


    def _split(self, X_column, split_value):
    
        left_idxs = []
        right_idxs = []

        # Loop on each value of the column to choose if it goes left or right
        for i in X_column.index.tolist():
            feature_value = X_column.loc[i]

            if ( isinstance(feature_value, float) and math.isnan(feature_value) ):
                right_idxs.append(i)    # If it's NaN, goes right as if it's false
                continue
            
            if self.splitting_test(feature_value, split_value):
                left_idxs.append(i)     # If it's true, goes left
            else:
                right_idxs.append(i)    # If it's false, goes right

        return left_idxs, right_idxs


    def splitting_function(self, X_column, y, threshold):

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 1
        
        left_impurity = self.splitting_criterion(y.loc[left_idxs])
        right_impurity = self.splitting_criterion(y.loc[right_idxs])
        result = (len(left_idxs) / len(y)) * left_impurity + (len(right_idxs) / len(y)) * right_impurity
        return result


    def predict(self, X):

        if isinstance(X, pd.DataFrame):
            return np.array([self.root.predict(X.loc[i]) for i in X.index])
        else:
            return self.root.predict(X)
        

    def print_tree(self, node=None, depth=0):
        
        if node is None:
            node = self.root
    
        if node.is_leaf:
            print(f"{'|   ' * depth}Predict: {node.label}")
        else:
            print(f"{'|   ' * depth}Feature {node.feature} < {node.threshold}?\t[{node.impurity}]")
        
            print(f"{'|   ' * (depth + 1)}Left:")
            self.print_tree(node.left, depth + 1)

            print(f"{'|   ' * (depth + 1)}Right:")
            self.print_tree(node.right, depth + 1)


    def evaluate(self, X, y):

        predictions = self.predict(X)

        loss = self._zero_one(predictions, y)
        precision = self._precision(predictions, y)
        recall = self._recall(predictions, y)

        return {'zero-one loss':loss, 'precision':precision, 'recall':recall}


    def _precision(self, predictions, y):

        all_positive = sum(predictions)

        tp = 0
        for i in range(0, len(predictions)):
            if (predictions[i] == 1 and predictions[i] == y[i]):
                tp += 1

        return tp/all_positive


    def _recall(self, predictions, y):

        tp = 0
        fn = 0
        for i in range(0, len(predictions)):
            if (predictions[i] == 1 and y[i] == 1):
                tp += 1
            elif (predictions[i] == 0 and y[i] == 1):
                fn += 1

        return tp/(tp+fn)


    def _zero_one(self, predictions, y):
        return np.mean(predictions != y)