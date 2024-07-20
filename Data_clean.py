import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the data
file_path = 'D:/Desktop/study in France/ESIGELEC-study/Intership/IPSOS/data_for_model_.xlsx'
data = pd.read_excel(file_path)


# Define the cleaning function
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    text = ' '.join(word for word in words if word not in stop_words)

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

    # Remove extra spaces
    text = text.strip()

    return text


# Apply the cleaning function to the dataset
data['cleaned_text'] = data['Original_verbatim'].apply(clean_text)

# Display the first few rows of the cleaned data
print(data.head())

# Save the cleaned data to a new file
cleaned_file_path = 'D:/Desktop/study in France/ESIGELEC-study/Intership/IPSOS/cleaned_data_for_model.xlsx'  # Specify the save path and filename
data.to_excel(cleaned_file_path, index=False)
