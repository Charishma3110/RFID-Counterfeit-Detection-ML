import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#Load the preprocessed dataset here
df = pd.read_csv(r"C:\Users\Cherry\OneDrive\Documents\CHS_project\Processed_RFID_Features_Windowed.csv")
X = df.drop(['PID', 'Time_Bin', 'Counterfeit'], axis=1)
y = df['Counterfeit']
X.fillna(X.mean(), inplace=True)

#Scaling is needed for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#Using ML models
models = {
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 150],
        'max_depth': [None, 10, 20]
    }),
    "SVM": (SVC(probability=True, class_weight='balanced', random_state=42), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }),
    "XGBoost": (XGBClassifier(eval_metric='logloss', random_state=42), {
        'n_estimators': [100, 150],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1]
    })
}
#Results
final_results = {}
print("\n Training with visuals...")

#Training the models
for name, (model, param_grid) in models.items():
    print(f"\n Training {name}...")
    X_input = X_scaled if name == "SVM" else X
    grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_input, y)
    best_model = grid.best_estimator_

    accs, precs, recs, f1s = [], [], [], []
    y_real, y_preds, y_probs = [], [], []
#Split train and test data
    for train_idx, test_idx in skf.split(X_input, y):
        if name == "SVM":
            X_train, X_test = X_input[train_idx], X_input[test_idx]
        else:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

        y_real.extend(y_test)
        y_preds.extend(y_pred)
        y_probs.extend(y_proba)

    final_results[name] = {
        "Accuracy": np.mean(accs),
        "Precision": np.mean(precs),
        "Recall": np.mean(recs),
        "F1-Score": np.mean(f1s),
        "Best Params": grid.best_params_
    }

    #Confusion Matrix
    cm = confusion_matrix(y_real, y_preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Counterfeit"],
                yticklabels=["Genuine", "Counterfeit"])
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    #ROC Curve
    fpr, tpr, _ = roc_curve(y_real, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    #Feature Importance for Random Forest
    if name == "Random Forest":
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances.sort_values().plot(kind='barh', title="Random Forest Feature Importance", figsize=(8, 6))
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()

# Print results summary
for model_name, results in final_results.items():
    print(f"\n {model_name} (5-Fold CV):")
    print(f"Accuracy : {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall   : {results['Recall']:.4f}")
    print(f"F1-Score : {results['F1-Score']:.4f}")
    print(f"Best Params: {results['Best Params']}")
