# Titanic Dataset Analytics

This analytics module performs exploratory data analysis on the Titanic dataset, examining survival patterns and relationships between variables.

## Data Preprocessing

- **Missing Value Handling**: Applied a threshold-based strategy where columns with <5% missing values were dropped, columns with 5-30% missing were imputed (using median for numeric, mode for categorical), and columns with >30% missing were filled with "missing" category.
- **Dataset Shape**: After preprocessing, the dataset contains the cleaned Titanic passenger records ready for analysis.

---

## Charts and Interpretations

### 1. Histogram of Numeric Columns (Age and Fare)

**Chart Description**: This histogram displays the distribution of continuous numeric variables including age and fare across all passengers.

**Interpretation**: The histogram reveals the distribution patterns of passenger demographics and ticket prices. Both age and fare columns show their respective frequency distributions, helping identify central tendencies and variability in passenger characteristics. This visualization is essential for understanding the range and concentration of values before conducting further statistical analysis.

---

### 2. Boxplot of Numeric Columns (Age and Fare)

**Chart Description**: The boxplot visualizes the central tendency, spread, and outliers for age and fare variables using quartiles and whiskers.

**Interpretation**: The boxplot effectively identifies outliers and the interquartile range (IQR) for both variables. Age data appears relatively compact with potential outliers at the extremes, while fare shows a wider distribution with several high-value outliers representing premium ticket prices. This visualization helps us understand data quality and which observations may warrant special attention in modeling.

---

### 3. Correlation Matrix Heatmap

**Chart Description**: A 6×6 correlation heatmap showing relationships between six key numeric variables: survived, pclass, age, sibsp, parch, and fare.

**Interpretation**: The heatmap reveals that **pclass and fare** show the strongest negative correlation (-0.56), indicating that higher class passengers (lower pclass numbers) paid higher fares. Additionally, **pclass and survived** are negatively correlated (-0.55), showing that passengers in first class were significantly more likely to survive than those in lower classes. These relationships suggest that economic status was a major factor in survival outcomes.

---

### 4. Survival Rate by Sex

**Chart Description**: A bar chart comparing the proportion of passengers who survived, broken down by gender (male/female).

**Interpretation**: Women had a substantially higher survival rate (approximately 74%) compared to men (approximately 19%), clearly demonstrating that the "women and children first" evacuation policy was strongly enforced during the Titanic disaster. This dramatic difference in survival rates by gender is one of the most striking patterns in the dataset and highlights the historical prioritization of women's safety during the evacuation.

---

### 5. Survival Rate by Passenger Class

**Chart Description**: A bar chart displaying survival rates across the three passenger classes (1st, 2nd, and 3rd).

**Interpretation**: First-class passengers had the highest survival rate (approximately 63%), followed by second-class passengers (approximately 47%), and third-class passengers had the lowest survival rate (approximately 24%). This stark hierarchy demonstrates that passengers' economic status and ticket class significantly influenced their chances of survival, likely due to better access to lifeboats and earlier evacuation information for wealthier passengers.

---

### 6. Survival Rate by Sex and Passenger Class (Combined Analysis)

**Chart Description**: A grouped bar chart showing survival rates for each combination of sex and passenger class, revealing the intersection of both demographic factors.

**Interpretation**: The intersectional analysis shows that women in first and second class had extremely high survival rates (over 90%), while third-class women had a lower but still substantial rate (~50%). In contrast, men across all classes had dramatically lower survival rates, with first-class men surviving at ~37%, second-class men at ~8%, and third-class men at ~16%. This combined view reveals that gender was the primary determinant of survival, but was heavily modulated by class, with privileged women far outlasting disadvantaged men.

---

## Key Findings Summary

1. **Gender was the strongest survival predictor**: Women were dramatically more likely to survive across all classes.
2. **Class status significantly affected survival**: First-class passengers had substantially better survival outcomes than lower classes.
3. **Intersectional disadvantage**: Third-class men faced the poorest survival odds, highlighting the compounding effects of low economic status and gender.
4. **Economic disparities in evacuation**: Better access to lifeboats and information for wealthy passengers clearly translated to survival advantages.
