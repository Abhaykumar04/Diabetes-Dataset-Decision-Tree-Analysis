

# Diabetes Dataset: Decision Tree Analysis

### I. Introduction

Explore the Pima Indians Diabetes dataset using Decision Tree models for both classification and regression. This project employs visualizations and analysis to understand key features and predictive outcomes related to diabetes.

### II. Data Exploration and Visualization

1. **Pairplot and Box Plot:**
   - Utilize seaborn pairplots to visualize feature relationships.
   - Box plots reveal distribution insights, aiding in outlier detection.

```python
sns.pairplot(df, hue='Outcome', diag_kind='kde')
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.drop(columns=['Outcome']), orient='h')
```
<img width="663" alt="image" src="https://github.com/Abhaykumar04/Diabetes-Dataset-Decision-Tree-Analysis/assets/112232080/0728927c-1c38-441a-af73-3ad8ab90e75b">


### III. Outlier Handling

1. **Outlier Capping:**
   - Mitigate outliers through capping using Interquartile Range (IQR).
   - Improve dataset integrity for robust modeling.

```python
df = handle_outliers(df, columns_to_check_outliers)
plt.figure(figsize=(12, 6))
sns.boxplot(df)
```
<img width="739" alt="image" src="https://github.com/Abhaykumar04/Diabetes-Dataset-Decision-Tree-Analysis/assets/112232080/fef3a09d-4d67-4d19-b8f6-a87223e0f88e">


### IV. Data Preparation and Splitting

1. **Feature-Target Split:**
   - Divide the data into features (X) and target (y).

```python
X = df.drop(columns=['Outcome'])
y = df['Outcome']
```


2. **Train-Test Split:**
   - Split the data into training and testing sets for model evaluation.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### V. Decision Tree Classification

1. **Model Training:**
   - Implement a Decision Tree Classifier with specified parameters.
   - Evaluate using a classification report and confusion matrix.

```python
clf = DecisionTreeClassifier(random_state=42, max_depth=3, max_features=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```
<img width="330" alt="image" src="https://github.com/Abhaykumar04/Diabetes-Dataset-Decision-Tree-Analysis/assets/112232080/f202e801-042b-4d3f-a1ff-cfac2f0ec577">


2. **Visualize the Decision Tree Classifier:**
   - Gain insights into decision-making processes within the classifier.

```python
plt.figure(figsize=(12, 6))
tree.plot_tree(clf, feature_names=column_names[:-1], class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title("Decision Tree Classifier")
plt.show()
```
<img width="737" alt="image" src="https://github.com/Abhaykumar04/Diabetes-Dataset-Decision-Tree-Analysis/assets/112232080/5cfc4b33-5aaf-4184-bf70-afcdb0916748">


### VI. Decision Tree Regression

1. **Model Training:**
   - Develop a Decision Tree Regressor with specified parameters.
   - Evaluate using Mean Squared Error (MSE).

```python
reg = DecisionTreeRegressor(random_state=42, max_depth=3, max_features=5)
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
print("Mean Squared Error (MSE):", mse)
```
<img width="255" alt="image" src="https://github.com/Abhaykumar04/Diabetes-Dataset-Decision-Tree-Analysis/assets/112232080/ab86aaef-58ce-46b8-a4ef-5c48dc798532">


2. **Visualize the Decision Tree Regressor:**
   - Illustrate the decision-making structure of the regressor.

```python
plt.figure(figsize=(12, 6))
tree.plot_tree(reg, feature_names=column_names[:-1], filled=True)
plt.title("Decision Tree Regressor")
plt.show()
```
<img width="739" alt="image" src="https://github.com/Abhaykumar04/Diabetes-Dataset-Decision-Tree-Analysis/assets/112232080/8299904d-af59-4491-8068-81648e448bcb">


Explore the intricacies of diabetes prediction through decision trees, gaining valuable insights for future model optimizations and healthcare applications.
