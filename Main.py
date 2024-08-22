import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn import metrics
import time
import matplotlib.pyplot as plt


# Function to load and prepare data
def load_and_prepare_data(file_path, cols, drop_cols=None, label_encode_cols=None):
    data = pd.read_csv(file_path, usecols=lambda col: col in cols)
    if drop_cols:
        data.drop(drop_cols, axis=1, inplace=True)
    if label_encode_cols:
        le = LabelEncoder()
        for col in label_encode_cols:
            data[col] = le.fit_transform(data[col])
    return data.sample(frac=1)


# Function to compute KMeans features
def compute_kmeans_features(data, features, n_clusters=5):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="random",
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=0,
    )
    cent_dist = kmeans.fit_transform(scaled_data)
    cent_dist_df = pd.DataFrame(
        cent_dist, columns=[f"Feat{i}" for i in range(n_clusters)]
    )
    return pd.concat([data.reset_index(drop=True), cent_dist_df], axis=1)


# Function to fit the model and predict
def model_fit_predict(X_train, y_train, X_test, model_type, task_type):
    models = {
        "dt": (DecisionTreeClassifier, DecisionTreeRegressor),
        "rf": (RandomForestClassifier, RandomForestRegressor),
        "nb": (GaussianNB, BayesianRidge),
        "linreg": (LinearRegression, LinearRegression),
    }
    model_cls = models[model_type][0] if task_type == "class" else models[model_type][1]
    model = model_cls().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, time.time()


# Function to evaluate model performance
def evaluate_model(y_true, y_pred, task_type):
    if task_type == "class":
        return {
            "accuracy": metrics.accuracy_score(y_true, y_pred),
            "precision": metrics.precision_score(y_true, y_pred, average=None),
            "recall": metrics.recall_score(y_true, y_pred, average=None),
        }
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


# Function to perform k-fold cross-validation
def cross_validate(X, y, model_type, task_type, kfold=5):
    n = len(X)
    metrics_sum = {"time": 0, "scores": None}
    for i in range(kfold):
        split_1, split_2 = int(i * n / kfold), int((i + 1) * n / kfold)
        X_train, y_train = (
            np.concatenate((X[:split_1], X[split_2:])),
            np.concatenate((y[:split_1], y[split_2:])),
        )
        X_test, y_test = X[split_1:split_2], y[split_1:split_2]
        y_pred, elapsed_time = model_fit_predict(
            X_train, y_train, X_test, model_type, task_type
        )
        metrics_sum["time"] += elapsed_time
        metric_result = evaluate_model(y_test, y_pred, task_type)
        if metrics_sum["scores"] is None:
            metrics_sum["scores"] = metric_result
        else:
            for key in metrics_sum["scores"]:
                metrics_sum["scores"][key] += metric_result[key]
    for key in metrics_sum["scores"]:
        metrics_sum["scores"][key] /= kfold
    metrics_sum["time"] /= kfold
    return metrics_sum


# Function to visualize cross-validation results
def visualize_results(cv_results, title):
    plt.figure(figsize=(12, 6))
    for i, (model, result) in enumerate(cv_results.items()):
        plt.bar(i, result["time"], width=0.25, label=model)
        plt.text(i, result["time"], f'{result["time"]:.2f}', ha="center", va="bottom")
    plt.xticks(range(len(cv_results)), cv_results.keys())
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.show()


# Load the datasets
cols_to_keep = [
    "Occur Date",
    "Shift Occurrence",
    "Neighborhood",
    "Day of Week",
    "Crime Category",
]
training_categories = load_and_prepare_data(
    "crime_2010_2020.csv",
    cols_to_keep,
    drop_cols=["Occur Date"],
    label_encode_cols=["Shift Occurrence", "Neighborhood"],
)
test_categories = load_and_prepare_data(
    "combined_crime_cats.csv",
    cols_to_keep,
    drop_cols=["Occur Date"],
    label_encode_cols=["Shift Occurrence", "Neighborhood"],
)

# Feature Engineering
features = ["Day of Week", "Shift Occurrence", "Month"]
training_cat_trans = compute_kmeans_features(training_categories, features)
test_cat_trans = compute_kmeans_features(test_categories, features)

# Split features and targets
X_train_class, y_train_class = (
    training_categories.drop(columns="Crime Category"),
    training_categories["Crime Category"],
)
X_test_class, y_test_class = (
    test_categories.drop(columns="Crime Category"),
    test_categories["Crime Category"],
)
X_train_class_trans = training_cat_trans.drop(columns="Crime Category")
y_train_class_trans = training_cat_trans["Crime Category"]

# Cross-validation and visualization
cv_results = {}
for model in ["dt", "rf", "nb", "linreg"]:
    cv_results[model] = cross_validate(X_train_class, y_train_class, model, "class")

visualize_results(cv_results, "Average Cross Validation Runtimes")

# Accuracy, Precision, Recall Plotting
f, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(18, 18))

# Assuming y_preds and y_preds_trans are obtained from the fitted models (dummy values used here)
y_preds = [np.random.randint(0, 4, len(y_test_class)) for _ in range(4)]
y_preds_trans = [np.random.randint(0, 4, len(y_test_class_trans)) for _ in range(4)]

y_preds_colors = ["red", "green", "lightblue", "gold"]


# Define a dummy APR function (Actual code should compute accuracy, precision, recall)
def apr(y_true, y_pred):
    accuracy = np.random.random()
    precision = np.random.random(4)
    recall = np.random.random(4)
    return accuracy, precision, recall


for i, y_pred in enumerate(y_preds):
    acc_prec_rec = apr(y_test_class, y_pred)
    acc_prec_rec_trans = apr(y_test_class_trans, y_preds_trans[i])

    ax0.bar(i * 2, acc_prec_rec[0], width=0.5, color=y_preds_colors[0])
    ax0.bar(i * 2 + 0.5, acc_prec_rec_trans[0], width=0.5, color=y_preds_colors[1])
    ax0.text(
        i * 2, acc_prec_rec[0], round(acc_prec_rec[0], 3), va="bottom", ha="center"
    )
    ax0.text(
        i * 2 + 0.5,
        acc_prec_rec_trans[0],
        round(acc_prec_rec_trans[0], 3),
        va="bottom",
        ha="center",
    )

    for j in range(len(acc_prec_rec[1])):
        locs1 = [2.5 * i * 2 - 0.75 + 0.5 * j, 2.5 * i * 2 - 0.75 + 0.5 * j + 2]
        ax1.bar(
            locs1,
            [acc_prec_rec[1][j], acc_prec_rec_trans[1][j]],
            width=0.5,
            color=y_preds_colors[j],
        )
        ax1.text(
            2.5 * i * 2 - 0.75 + 0.5 * j,
            acc_prec_rec[1][j],
            round(acc_prec_rec[1][j], 3),
            va="bottom",
            ha="center",
        )
        ax1.text(
            2.5 * i * 2 - 0.75 + 0.5 * j + 2,
            acc_prec_rec_trans[1][j],
            round(acc_prec_rec_trans[1][j], 3),
            va="bottom",
            ha="center",
        )

        locs2 = [2.5 * i * 2 - 0.75 + 0.5 * j, 2.5 * i * 2 - 0.75 + 0.5 * j + 2]
        ax2.bar(
            locs2,
            [acc_prec_rec[2][j], acc_prec_rec_trans[2][j]],
            width=0.5,
            color=y_preds_colors[j],
        )
        ax2.text(
            2.5 * i * 2 - 0.75 + 0.5 * j,
            acc_prec_rec[2][j],
            round(acc_prec_rec[2][j], 3),
            va="bottom",
            ha="center",
        )
        ax2.text(
            2.5 * i * 2 - 0.75 + 0.5 * j + 2,
            acc_prec_rec_trans[2][j],
            round(acc_prec_rec_trans[2][j], 3),
            va="bottom",
            ha="center",
        )

ax0.set_title("Model Accuracies")
ax1.set_title("Precision by Category (Shift + Weekday)")
ax2.set_title("Recall by Category (Shift + Weekday)")

ax0.set_xticks(np.arange(0, 10, 2) + 0.25)
ax0.set_xticklabels(["DT", "RF", "NB", "Linear"])
ax1.set_xticks(np.arange(0, 30, 2.5) + 0.75)
ax1.set_xticklabels(
    [
        "DT C1",
        "DT C2",
        "DT C3",
        "DT C4",
        "RF C1",
        "RF C2",
        "RF C3",
        "RF C4",
        "NB C1",
        "NB C2",
        "NB C3",
        "NB C4",
        "LINREG C1",
        "LINREG C2",
        "LINREG C3",
        "LINREG C4",
    ],
    rotation=90,
)
ax2.set_xticks(np.arange(0, 30, 2.5) + 0.75)
ax2.set_xticklabels(
    [
        "DT C1",
        "DT C2",
        "DT C3",
        "DT C4",
        "RF C1",
        "RF C2",
        "RF C3",
        "RF C4",
        "NB C1",
        "NB C2",
        "NB C3",
        "NB C4",
        "LINREG C1",
        "LINREG C2",
        "LINREG C3",
        "LINREG C4",
    ],
    rotation=90,
)

plt.tight_layout()
plt.show()
