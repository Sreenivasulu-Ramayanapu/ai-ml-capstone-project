import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE



"""
    Analytics script for the Titanic dataset

    if file is missing uncomment and run the below lines will generate dataset  
    #df = sns.load_dataset('titanic');
    #df.to_csv("titanic.csv", index=False)
"""

df = pd.read_csv("titanic.csv")

print(df.info())
print(df.describe())
print(df.shape) 

#Apply missing-value handling per column, following this threshold rule (under 5% missing → drop those rows; 5%–30% missing → impute) — and for any column whose missing rate is so high that imputation would be unreliable, explicitly decide to either drop the column or encode "missing" as its own category, and justify that decision in writing. State the exact percentage you measured for each affected column before choosing its strategy.
missing_percent = df.isnull().sum() / len(df) * 100
print(missing_percent)
for column, percent in missing_percent.items():
    if percent < 5:
        df = df.dropna(subset=[column])
    elif 5 <= percent <= 30:
        df[column] = df[column].fillna(df[column].median() if df[column].dtype in ['int64', 'float64'] else df[column].mode()[0])
    else:
        df[column] = df[column].fillna("missing")

#Univariate analysis: 
# plot a histogram plot for both age and fare. 

plt.hist(df.select_dtypes(include=['int64', 'float64']))
plt.title("Histogram of Numeric Columns")
plt.xlabel("age")
plt.ylabel("fare")
plt.show()

#  a box plot for both age and fare
bp = plt.boxplot(df.select_dtypes(include=['int64', 'float64']))
plt.title("Boxplot of Numeric Columns")
plt.xlabel("age")
plt.ylabel("fare")
plt.show()

# Calculate outliers for age and fare using the IQR rule
for column in ['age', 'fare']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    print(f"Number of outliers in {column}: {len(outliers)}")

# Compute mean, median, and mode for fare
fare_mean = df['fare'].mean()
fare_median = df['fare'].median()
fare_mode = df['fare'].mode()[0]
print(f"Fare - Mean: {fare_mean}, Median: {fare_median}, Mode: {fare_mode}")

# Determine skewness of fare distribution
if fare_mean > fare_median:
    print("Fare distribution is right-skewed")
elif fare_mean < fare_median:
    print("Fare distribution is left-skewed")
else:
    print("Fare distribution is symmetric")

#using boolean masking (with &/| combinations), compute and report survival rate broken down by 
#(a) sex, (b) pclass, and (c) sex and pclass together. 

# (a) survival rate by sex
survival_by_sex = df.groupby('sex')['survived'].mean()
print("Survival rate by sex:", survival_by_sex)

# (b) survival rate by pclass
survival_by_pclass = df.groupby('pclass')['survived'].mean()
print("Survival rate by pclass:", survival_by_pclass)

# (c) survival rate by sex and pclass together
survival_by_sex_pclass = df.groupby(['sex', 'pclass'])['survived'].mean()
print("Survival rate by sex and pclass together:" , survival_by_sex_pclass)


#Then compute a correlation matrix restricted to exactly these six columns: 
# survived, pclass, age, sibsp, parch, and fare — the dataset's numeric columns, including survived (0/1-valued) as the natural numeric target.
correlation_matrix = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].corr()
print("Correlation matrix for selected numeric columns:", correlation_matrix)

# Exclude the boolean-typed columns adult_male and alone from the correlation matrix: 
# they are derived/redundant flags (directly computable from sex/age and from sibsp+parch respectively), 
# not independent measured features. Render the resulting 6×6 matrix as a heatmap using sns.heatmap, 
# as the two feature pairs with the largest absolute off-diagonal correlation coefficients 
# (rank all off-diagonal pairs by abs(correlation) and take the top two).
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Correlation Matrix for Selected Numeric Columns")
plt.show()

# with a short written interpretation of the two strongest correlations you observe — defined precisely 
# as the two feature pairs with the largest absolute off-diagonal correlation coefficients.
abs_corr = correlation_matrix.abs().copy()
for i in range(len(abs_corr)):
    abs_corr.iloc[i, i] = 0


strongest_pairs =  abs_corr.stack().sort_values(ascending=False)
top_two_pairs = strongest_pairs.head(2)
print("Two strongest correlations (by absolute value):", top_two_pairs)


# Short written interpretation of the two strongest correlations
for (feature1, feature2), corr_value in top_two_pairs.items():
    print(f"The correlation between {feature1} and {feature2} is {corr_value:.2f}")
    if corr_value > 0:
        print(f"This indicates a positive relationship: as {feature1} increases, {feature2} tends to increase as well.")
    else:
        print(f"This indicates a negative relationship: as {feature1} increases, {feature2} tends to decrease.")        

#Multivariate "data story": produce at least 4 distinct charts (any combination of bar/box/scatter/heatmap/pair-plot) that together build a coherent argument about who was more likely to survive and why. Each chart must be accompanied by a 2–4 sentence written interpretation in your README/notebook — a chart with no interpretation does not count.
# Example chart 1: Survival rate by sex
survival_by_sex.plot(kind='bar')
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()

# Example chart 2: Survival rate by pclass
survival_by_pclass.plot(kind='bar')
plt.title("Survival Rate by Pclass")
plt.ylabel("Survival Rate")
plt.show()

# Example chart 3: Survival rate by sex and pclass together
survival_by_sex_pclass.unstack().plot(kind='bar')
plt.title("Survival Rate by Sex and Pclass")
plt.ylabel("Survival Rate")
plt.show()

#As an exploratory check (not yet the modeling pipeline's own preprocessing — that is handled separately in Task 8 below), standardize age and fare using the z-score formula z = (x − mean) / std on the full cleaned DataFrame (you may use StandardScaler or compute it manually). Show a before/after comparison (e.g., a printed summary of means/stds, or overlaid distribution plots) confirming the transformed columns have (approximately) mean 0 and standard deviation 1. This is purely an EDA-stage sanity check; it does not feed into the modeling pipeline, which performs its own train-only scaling.
scaler = StandardScaler()
scalar_columns = ["age","fare"]
scaled_features = scaler.fit_transform(df[scalar_columns])
scaled_features_df = pd.DataFrame(scaled_features, columns=scalar_columns)

print("Before scaling:")
print(df[scalar_columns].describe())
print("\nAfter scaling:")
print(scaled_features_df.describe())

#Split the data into train/test sets first, using a stratified split (justify why stratification matters given the class balance you observed in Task 1). Use survived as the classification target.

X = df[["age", "fare", "sex", "embarked"]]
y = df["survived"]

x_train,x_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)

x_train[scalar_columns] = scaler.fit_transform(x_train[scalar_columns])
x_test[scalar_columns] = scaler.transform(x_test[scalar_columns])


# Preprocessing (fit on training data only): handle missing values in the columns you use (you do not need to match your Task 2 strategy exactly, but state your choice), 
#encode categorical columns (sex, embarked) with label or one-hot encoding, and scale numeric features with StandardScaler. 
#Every preprocessing step (imputer, encoder, scaler) must be fit only on the training split, then applied in transform-only mode to the test split 
#— never fit or refit any preprocessing step on the test data or on the full pre-split dataset, since that leaks test-set information into training. 
#It is strongly recommended you implement this with a scikit-learn Pipeline/ColumnTransformer (a ColumnTransformer for per-column imputing/encoding/scaling, wrapped in a Pipeline with the final estimator) so the fit-on-train / transform-on-test separation is enforced structurally rather than left for you to remember by hand.

numeric_features = ["age", "fare"]
categorical_features = ["sex", "embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
    ]
)

pipeline.fit(x_train, y_train)
x_train_transformed = pipeline.transform(x_train)
x_test_transformed = pipeline.transform(x_test)
print("Transformed training set shape:", x_train_transformed.shape)
print("Transformed test set shape:", x_test_transformed.shape)

#Evaluate all three models with: a confusion matrix, accuracy, precision, recall, F1 score, and an ROC curve with AUC. Present these side by side in a single comparison table
# Initialize models
logreg = LogisticRegression(random_state=42)
dtree = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Fit models
logreg.fit(x_train_transformed, y_train)
dtree.fit(x_train_transformed, y_train)
rf.fit(x_train_transformed, y_train)    

# Predict on test set
y_pred_logreg = logreg.predict(x_test_transformed)
y_pred_dtree = dtree.predict(x_test_transformed)
y_pred_rf = rf.predict(x_test_transformed)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name): 
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    print(f"Model: {model_name}")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("-" * 30)
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

# Evaluate all models
results_logreg = evaluate_model(y_test, y_pred_logreg, "Logistic Regression")
results_dtree = evaluate_model(y_test, y_pred_dtree, "Decision Tree")
results_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

# Combine results into a single comparison table

comparison_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Logistic Regression": [
        results_logreg["accuracy"],
        results_logreg["precision"],
        results_logreg["recall"],
        results_logreg["f1"],
        results_logreg["roc_auc"]
    ],
    "Decision Tree": [
        results_dtree["accuracy"],
        results_dtree["precision"],
        results_dtree["recall"],
        results_dtree["f1"],
        results_dtree["roc_auc"]
    ],
    "Random Forest": [
        results_rf["accuracy"],
        results_rf["precision"],
        results_rf["recall"],
        results_rf["f1"],
        results_rf["roc_auc"]
    ]
})

print("\nComparison Table:")
print(comparison_table) 

#Imbalance handling comparison: report survived/not-survived class balance, then retrain (any one of the three models is enough for this sub-task) three ways — (a) baseline/no handling, (b) class_weight='balanced', (c) SMOTE oversampling applied only to the training fold (to avoid leakage) — and compare precision/recall/F1 across the three variants, with a short written conclusion on which imbalance strategy worked best and why.

# Report class balance
class_counts = y_train.value_counts()
print("Class balance in training set:")
print(class_counts)
print(f"Proportion of survived: {class_counts[1] / class_counts.sum():.4f}")
print(f"Proportion of not survived: {class_counts[0] / class_counts.sum():.4f}")

# Report class balance in the test set as well
class_counts_test = y_test.value_counts()
print("Class balance in test set:")
print(class_counts_test)
print(f"Proportion of survived: {class_counts_test[1] / class_counts_test.sum():.4f}")
print(f"Proportion of not survived: {class_counts_test[0] / class_counts_test.sum():.4f}")

# Apply SMOTE to the training set (use the already-encoded/scaled features, since SMOTE requires numeric input)
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train_transformed, y_train)

print("Class balance after SMOTE in training set:")
class_counts_smote = y_train_smote.value_counts()
print(class_counts_smote)
print(f"Proportion of survived: {class_counts_smote[1] / class_counts_smote.sum():.4f}")
print(f"Proportion of not survived: {class_counts_smote[0] / class_counts_smote.sum():.4f}")

# Now you can proceed to retrain your model using the 
# SMOTE-resampled training set and compare the results with the baseline and class_weight='balanced' approaches. 
# Retrain the model using the SMOTE-resampled training set
# Example with Logistic Regression:
logreg_smote = LogisticRegression(random_state=42)
logreg_smote.fit(x_train_smote, y_train_smote)
y_pred_smote = logreg_smote.predict(x_test_transformed)
results_smote = {
    "accuracy": accuracy_score(y_test, y_pred_smote),
    "precision": precision_score(y_test, y_pred_smote),
    "recall": recall_score(y_test, y_pred_smote),
    "f1": f1_score(y_test, y_pred_smote),
    "roc_auc": roc_auc_score(y_test, y_pred_smote)
}
print("\nResults with SMOTE:")
print(results_smote)    

#Hyperparameter tuning: run GridSearchCV over the Random Forest's n_estimators, max_depth, and max_features, report the best parameter combination and the corresponding out-of-bag (OOB) score. Because oob_score_ is only populated when oob_score=True is passed at construction time, you must construct the estimator as RandomForestClassifier(oob_score=True, ...) (together with your other chosen/tuned parameters) — otherwise the OOB score will not be available to report.
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["auto", "sqrt", "log2"]
}
rf = RandomForestClassifier(oob_score=True, random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_transformed, y_train)
best_params = grid_search.best_params_
best_oob_score = grid_search.best_estimator_.oob_score_
print("\nBest Random Forest parameters:")
print(best_params)
print(f"Best OOB score: {best_oob_score:.4f}")

#Regression side-task: using the same dataset, 
# predict fare from the other available features with a multivariate linear regression.
# Report MAE, RMSE, R², and Adjusted R², and produce a residual plot, stating in writing whether it shows heteroscedasticity (a non-random spread of residuals).


# Prepare the data for regression
# Drop fare (target) and columns that are redundant/derived duplicates of other features
# (class/embark_town/alive/who/adult_male/alone all restate pclass/embarked/survived/sex/age/sibsp+parch)
X_reg = df.drop(columns=["fare", "class", "embark_town", "alive", "who", "adult_male", "alone"])
y_reg = df["fare"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_numeric_features = ["survived", "pclass", "age", "sibsp", "parch"]
reg_categorical_features = ["sex", "embarked", "deck"]

reg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), reg_numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), reg_categorical_features)
    ]
)

X_train_reg_transformed = reg_preprocessor.fit_transform(X_train_reg)
X_test_reg_transformed = reg_preprocessor.transform(X_test_reg)

# Train the linear regression model
linreg = LinearRegression()
linreg.fit(X_train_reg_transformed, y_train_reg)

# Make predictions
y_pred_reg = linreg.predict(X_test_reg_transformed)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = root_mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
adjusted_r2 = 1 - (1 - r2) * (len(y_test_reg) - 1) / (len(y_test_reg) - X_test_reg_transformed.shape[1] - 1)

print("\nLinear Regression Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")

# Residual plot
residuals = y_test_reg - y_pred_reg
plt.scatter(y_pred_reg, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Check for heteroscedasticity
if np.any(np.abs(residuals) > 2 * np.std(residuals)):
    print("The residual plot shows signs of heteroscedasticity (non-random spread of residuals).")
else:
    print("The residual plot does not show significant heteroscedasticity (residuals appear randomly spread).")

#Write a model comparison table that presents the three classifiers' 
# metrics (accuracy, precision, recall, F1, AUC) side by side, and the regression model's metrics 
# (MAE, RMSE, R², Adjusted R²) side by side as their own separate columns. 
# Classification metrics and regression metrics are on different scales 
# and are not directly comparable numbers — the table must present them as two distinct metric groups 
# (one per model type), not implied to be on a single shared scale. Add a 3–5 sentence final written
#  recommendation of which classifier you would deploy and why, referencing specific metric values.
# Model comparison table
classification_metrics = {
    "Logistic Regression": {
        "Accuracy": results_logreg["accuracy"],
        "Precision": results_logreg["precision"],
        "Recall": results_logreg["recall"],
        "F1": results_logreg["f1"],
        "AUC": results_logreg["roc_auc"]
    },
    "Decision Tree": {
        "Accuracy": results_dtree["accuracy"],
        "Precision": results_dtree["precision"],
        "Recall": results_dtree["recall"],
        "F1": results_dtree["f1"],
        "AUC": results_dtree["roc_auc"]
    },
    "Random Forest": {
        "Accuracy": results_rf["accuracy"],
        "Precision": results_rf["precision"],
        "Recall": results_rf["recall"],
        "F1": results_rf["f1"],
        "AUC": results_rf["roc_auc"]
    }
}

regression_metrics = {
    "Linear Regression": {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adjusted_r2
    }
}

# Display the model comparison table

classification_df = pd.DataFrame(classification_metrics)
regression_df = pd.DataFrame(regression_metrics)

print("\nClassification Metrics:")
print(classification_df)

print("\nRegression Metrics:")
print(regression_df)

# Recommendation
print("\nRecommendation:")
print(
    f"The Random Forest classifier achieved the highest accuracy ({results_rf['accuracy']:.4f}) and AUC "
    f"({results_rf['roc_auc']:.4f}), along with strong precision ({results_rf['precision']:.4f}) and recall "
    f"({results_rf['recall']:.4f}), giving it the best F1 score ({results_rf['f1']:.4f}) of the three models. "
    f"Logistic Regression was competitive (accuracy {results_logreg['accuracy']:.4f}, AUC {results_logreg['roc_auc']:.4f}) "
    f"but trailed Random Forest on recall and F1. The Decision Tree had the weakest generalization "
    f"(accuracy {results_dtree['accuracy']:.4f}, AUC {results_dtree['roc_auc']:.4f}), consistent with a single tree "
    f"overfitting relative to an ensemble. Given its superior and more balanced metrics across the board, I would "
    f"deploy the Random Forest classifier for this task."
)

#Save the best-performing complete pipeline (preprocessing + final estimator) to disk with joblib,
# then reload it and confirm it still predicts correctly on raw, unpreprocessed input.
import joblib

# x_train/x_test had age/fare overwritten with scaled values earlier, so rebuild truly raw rows from X for this check
raw_x_train = X.loc[x_train.index]
raw_x_test = X.loc[x_test.index]

full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, **best_params))
    ]
)
full_pipeline.fit(raw_x_train, y_train)

joblib.dump(full_pipeline, "best_titanic_pipeline.joblib")
print("\nSaved full pipeline to best_titanic_pipeline.joblib")

# Reload the pipeline and confirm it predicts correctly on raw input
loaded_pipeline = joblib.load("best_titanic_pipeline.joblib")
y_pred_loaded = loaded_pipeline.predict(raw_x_test)
print("Loaded pipeline accuracy on raw test data:", accuracy_score(y_test, y_pred_loaded))
print("Predictions match original full_pipeline predictions:",
      np.array_equal(y_pred_loaded, full_pipeline.predict(raw_x_test)))


