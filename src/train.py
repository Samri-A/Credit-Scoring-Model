import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
from data_processing import my_pipeline , load_data
import mlflow
import mlflow.sklearn
import os

def train():
    mlflow.set_experiment("credit-risk-model")


    path = "./data/raw/training.csv"
    df = load_data(path)
    processed_data = my_pipeline.fit_transform(df)
    feature_names = my_pipeline.named_steps['columntransformer'].get_feature_names_out()
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    categorical_features = "cat__is_high_risk"
    processed_df[categorical_features] = LabelEncoder().fit_transform(processed_df[categorical_features])
    X = processed_df.drop(columns="cat__is_high_risk")
    y = processed_df["cat__is_high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                                    
    with mlflow.start_run(run_name="RandomForest"):
       clf = RandomForestClassifier(random_state=42)
   
       base_mean = processed_df["cat__is_high_risk"].value_counts(normalize=True).max()
       mlflow.log_metric("Baseline_Accuracy", base_mean)
   
       # Cross-validation
       cv_acc_scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
       mlflow.log_metric("CV_Accuracy_Mean", cv_acc_scores.mean())
   
       params = {
           "n_estimators": range(25, 100, 25),
           "max_depth": range(10, 50, 10)
       }
   
       grid_model = GridSearchCV(
           clf,
           param_grid=params,
           cv=5,
           n_jobs=-1,
           verbose=1
       )
       grid_model.fit(X_train, y_train)
   
       mlflow.log_params(grid_model.best_params_)
       best_clf = grid_model.best_estimator_
   
       # ⬇️ TEST metrics
       y_pred = best_clf.predict(X_test)
       y_proba = best_clf.predict_proba(X_test)[:, 1]
   
       acc_rf = accuracy_score(y_test, y_pred)
       prec_rf = precision_score(y_test, y_pred)
       rec_rf = recall_score(y_test, y_pred)
       f1_rf = f1_score(y_test, y_pred)
       roc_rf = roc_auc_score(y_test, y_proba)
   
       mlflow.log_metric("test_accuracy", acc_rf)
       mlflow.log_metric("test_precision", prec_rf)
       mlflow.log_metric("test_recall", rec_rf)
       mlflow.log_metric("test_f1_score", f1_rf)
       mlflow.log_metric("test_roc_auc", roc_rf)
   
       # ⬇️ TRAIN metrics
       y_train_pred = best_clf.predict(X_train)
       y_train_proba = best_clf.predict_proba(X_train)[:, 1]
   
       mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
       mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred))
       mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred))
       mlflow.log_metric("train_f1_score", f1_score(y_train, y_train_pred))
       mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, y_train_proba))
   
       mlflow.sklearn.log_model(best_clf, "best_random_forest_model")
       mlflow.set_tag("Model_Type", "RandomForest + GridSearch")
       mlflow.set_tag("Data", "Training data from ../data/raw/training.csv")
    
       print("✅ Random forest complete. Best params logged.")
   
    with mlflow.start_run(run_name="LogisticRegression", nested=True):
       log_grid = {
           "C": [0.001, 0.01, 0.1, 1.0, 10.0],
           "penalty": ["l1", "l2"],
           "solver": ["liblinear", "saga"]
       }
   
       log_model = GridSearchCV(LogisticRegression(max_iter=1000), log_grid, cv=5, scoring="f1", n_jobs=-1)
       log_model.fit(X_train, y_train)
   
       log_best = log_model.best_estimator_
   
       # ⬇️ TEST metrics
       y_pred_log = log_best.predict(X_test)
       y_proba_log = log_best.predict_proba(X_test)[:, 1]
   
       mlflow.log_params(log_model.best_params_)
       mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_log))
       mlflow.log_metric("test_precision", precision_score(y_test, y_pred_log))
       mlflow.log_metric("test_recall", recall_score(y_test, y_pred_log))
       mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_log))
       mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba_log))
   
       # ⬇️ TRAIN metrics
       y_train_pred_log = log_best.predict(X_train)
       y_train_proba_log = log_best.predict_proba(X_train)[:, 1]
   
       mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred_log))
       mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred_log))
       mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred_log))
       mlflow.log_metric("train_f1_score", f1_score(y_train, y_train_pred_log))
       mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, y_train_proba_log))
   
       mlflow.sklearn.log_model(log_best, "logistic_regression_model")
   
       print("✅ Logistic regression complete. Best params logged.")


    
    
    # create a folder for your models
    output_dir = "/Users/HP/Desktop/Tenx/Credit-Scoring-Model/models"
    os.makedirs(output_dir, exist_ok=True)
    
    # after training & grid search...
    best_rf = grid_model.best_estimator_
    best_log = log_model.best_estimator_
    
    # save RandomForest locally
    mlflow.sklearn.save_model(
        sk_model=best_rf,
        path=os.path.join(output_dir, "random_forest_model")
    )
    
    # save LogisticRegression locally
    mlflow.sklearn.save_model(
        sk_model=best_log,
        path=os.path.join(output_dir, "logistic_regression_model")
    )

if __name__ == "__main__":
       train()
   
   