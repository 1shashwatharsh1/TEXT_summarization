import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))  # Load English stop words
    words = text.split()  # Split the text into words
    processed_words = [word for word in words if word.lower() not in stop_words]  # Remove stop words
    return ' '.join(processed_words)  # Join the words back into a single string

# Load the dataset
data = pd.read_csv('data/impression_data.csv')  # Adjust this to your actual dataset name

# Print the column names to ensure everything is correct
print("Column names in the dataset:")
print(data.columns.tolist())

# Combine 'History' and 'Observation' for processing
data['processed_text'] = (data['History'] + ' ' + data['Observation']).apply(preprocess_text)

# If needed, you can also preprocess the 'Impression' column
data['processed_impression'] = data['Impression'].apply(preprocess_text)

# Optionally, display the first few rows of the processed DataFrame
print("Processed data preview:")
print(data[['processed_text', 'processed_impression']].head())

# Save the processed data to a new CSV file
data.to_csv('data/processed_data.csv', index=False)  # Save the processed data
