import numpy as np
import pandas as pd
from TreeClassifier import TreeClassifier

def cross_validation(tree, X: pd.DataFrame, y, K: int = 3):
	
    if tree is None or not isinstance(tree, TreeClassifier):
        raise TypeError("'tree' must be instance of 'TreeClassifier'")

    if isinstance(y, np.ndarray):
        Y = y
    else:
        Y = np.array(y, dtype=int)

    df = X.copy()
    df = df.reset_index(drop=True)

    indices = np.random.permutation(len(df))

    fold_size = len(df) // K
    print("Fold size: ", fold_size)

    m_loss, m_precision, m_recall = [], [], []

    for k in range(K):
        # Define the index for train and test
        test_indices = indices[k*fold_size:(k+1)*fold_size]
        train_indices = np.concatenate([indices[:k*fold_size], indices[(k+1)*fold_size:]])

        # Split the datas
        X_train, X_test = df.loc[train_indices], df.loc[test_indices]
        y_train, y_test = Y[train_indices], Y[test_indices]
    
        # Train the tree
        tree.fit(X_train, y_train)
    
        # Evaluate the trained tree
        loss = tree.evaluate(X_test, y_test)
        m_loss.append(loss["zero-one loss"])
        m_precision.append(loss["precision"])
        m_recall.append(loss["recall"])
    
    m_loss = np.mean(m_loss)
    m_precision = np.mean(m_precision)
    m_recall = np.mean(m_recall)
    return {'zero-one loss':m_loss, 'precision':m_precision, 'recall':m_recall}