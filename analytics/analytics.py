import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
