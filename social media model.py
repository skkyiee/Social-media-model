import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and validation datasets
training_file_path = 'twitter_training.csv'
validation_file_path = 'twitter_validation.csv'

training_data = pd.read_csv(training_file_path)
validation_data = pd.read_csv(validation_file_path)

# Combine the datasets
combined_data = pd.concat([training_data, validation_data], ignore_index=True)

# Drop rows with missing comments
combined_data_cleaned = combined_data.dropna(subset=['comment'])

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(combined_data_cleaned['comment'])
y_train = combined_data_cleaned['feeling']

# Split validation data into features and labels
X_valid = vectorizer.transform(validation_data['comment'].fillna(''))
y_valid = validation_data['feeling']

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_valid)

# Evaluate the model
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
