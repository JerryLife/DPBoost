import pandas as pd
import numpy as np
import copy
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
# Load dataset from data/titanic
dataset_name = "adult"
label_name = "salary>50K"
data_train = pd.read_csv(f"data/{dataset_name}/clean/{dataset_name}_train_onehot.csv")
data_test = pd.read_csv(f"data/{dataset_name}/clean/{dataset_name}_test_onehot.csv")

# Split the dataset into training and validation sets
X_train = data_train.drop(columns=[label_name])
y_train = data_train[label_name].astype(int)
X_test = data_test.drop(columns=[label_name])
y_test = data_test[label_name].astype(int)

# n_classes = len(set(y_train))

# Map train labels to new class labels
# First, get unique labels and create a mapping
unique_labels = sorted(set(y_train))
label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

# ensure y_test labels are in the mapping
assert all(label in label_map for label in y_test)

# Apply the mapping to train and test labels
y_train = y_train.map(label_map)
y_test = y_test.map(label_map)

# Change 0 label to -1
y_train = y_train.map(lambda x: (x - 0.5) * 2)
y_test = y_test.map(lambda x: (x - 0.5) * 2)

params = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'tree_learner': 'serial',
    'learning_rate': 0.1,
    'max_bin': 255,
    'num_leaves': 32,
    'min_data_in_leaf': 1,
    'max_depth': 3,
    'verbose': 2,
    'num_iterations': 20,
    'total_budget': 2,
    'geo_clip': 1,
    'high_level_boost_round': 1,
    'inner_boost_round': 20,
    'boost_method': 'DPBoost_2level',
}


# Create dataset for LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Train the model
params_copy = copy.deepcopy(params)
model = lgb.train(
    params_copy,
    lgb_train,
    num_boost_round=20,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=1
)

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = ((y_pred_prob > 0).astype(int) - 0.5) * 2


# Evaluate the model
accuracy = np.isclose(y_pred, y_test, atol=1e-6).mean()
print(f"Accuracy: {accuracy:.4f}")

auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.4f}")


