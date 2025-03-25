import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def evaluate(model, X_test, y_true, classes, test_idx):

    y_pred = model.predict(X_test)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    # COnfusion matrix
    mat = confusion_matrix(y_true, y_pred)
    df_mat = pd.DataFrame(mat, columns=classes)
    df_mat = df_mat.set_index([classes])
    sns.heatmap(df_mat, annot=True)

    # flour prices per category
    prices = [1.2, 2.5, 5]
    # Number of sold samples per category
    sold = mat.sum(axis=0)
    # Initial debt, ie. Caroline spent len(y_test) * 1CHF.
    benefit = -1 * len(y_true)
    cost = 0
    for i in range(3):
        # Caroline sold everything
        benefit = benefit + sold[i] * prices[i]
        for j in range(3):
            if j != i:
                # Caroline must refund the wrong orders
                cost = cost + mat[i,j]*prices[j]
    print(f"Total benefit {benefit -cost :.4f} CHF on {len(y_true)} examples.")

    # Print scores
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred)}")
    print(f"Macro f1-score: {f1_score(y_true, y_pred, average='macro')}")
    print(f"Macro precision: {precision_score(y_true, y_pred, average='macro')}")
    print(f"Macro recall: {recall_score(y_true, y_pred, average='macro')}")
    print(f"Kappa score: {cohen_kappa_score(y_true, y_pred)}")
    print(classification_report(y_true, y_pred, target_names=classes))

    # Compute misclassified samples
    df_misclassified = pd.DataFrame()
    df_misclassified['True label'] = y_true
    df_misclassified['ID'] = test_idx
    df_misclassified['Predicted label'] = y_pred
    miscl_samples = df_misclassified.loc[df_misclassified['True label'] != df_misclassified['Predicted label'], 'ID'].to_list()

    return miscl_samples, benefit

def prepare_dataset(df : pd.DataFrame, label_dim: int = 1, test_size=0.2, val_size=None, rd=42):

    assert(label_dim in [1,2])

    if label_dim == 2:
        y = pd.get_dummies(df['Quality'], columns=['Quality']).astype(int)
    else:
        y = df['Quality']
    X = df.drop('Quality', axis=1)
    features = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rd, stratify=y)
    test_idx = X_test.index

    if val_size is not None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=rd, stratify=y_train)
        train_idx = X_train.index
        val_idx = X_val.index

        return X_train, X_test, X_val, y_train, y_test, y_val, train_idx, test_idx, val_idx, features

    train_idx = X_train.index
    return X_train, X_test, y_train, y_test, train_idx, test_idx, features



