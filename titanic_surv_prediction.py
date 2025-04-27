# Importing of libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading of dataset
df = pd.read_csv("titanic.csv")

# data cleaning

# filling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Converting categorical into numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Selecting features and targets 
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# training data

# Splitting data into training/testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluating models

# Make predictions
y_pred = model.predict(X_test)

# Show accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

#custom predictions here

# Format: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
example_passenger = np.array([[3, 0, 22, 1, 0, 7.25, 0]])  # 3rd class male, 22yo
result = model.predict(example_passenger)

print("Survived?" , "Yes 🎉" if result[0] == 1 else "No")
