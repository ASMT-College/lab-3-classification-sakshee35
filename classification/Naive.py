import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report

# Step 1: Load the dataset
df = pd.read_csv('employee_data.csv')
# Step 2: Split the data into features and target
X = df.drop(columns='Outcome')
y = df['Outcome']
# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)
# Step 4: Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()
# Step 5: Train the model
nb_classifier.fit(X_train, y_train)
# Step 6: Make predictions
y_pred_nb = nb_classifier.predict(X_test)
# Step 7: Evaluate the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")
print("\nClassification Report:\n", classification_report(y_test,
y_pred_nb))