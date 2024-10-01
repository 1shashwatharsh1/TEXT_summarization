from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load nltk resources
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load model and token
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Using a pre-trained sentence transformer for embeddings
token = os.getenv("HUGGINGFACE_TOKEN")  # Ensure your token is stored in the .env file

# Load the dataset
data = pd.read_csv('data/impression_data.csv')  # Adjust to your actual dataset

# Preprocess text
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

data['processed_text'] = data['input_text'].apply(preprocess_text)

# Convert to embeddings
model = SentenceTransformer(model_name)
embeddings = model.encode(data['processed_text'].tolist())

# Identify top 100 pairs of similar words
similarity_matrix = cosine_similarity(embeddings)
# Code to find top 100 word pairs based on similarity_matrix goes here

# Save or visualize the results as needed
