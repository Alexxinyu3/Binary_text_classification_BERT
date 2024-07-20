import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

# Load the BERT model from pth file
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Load the model state_dict
model.load_state_dict(torch.load('output/weights/best.pth'))

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Define a function to predict the sentiment of a text
def predict_sentiment(text):
    # Tokenize and encode the text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move the inputs to the GPU
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted sentiment
    prediction = outputs.logits.item()
    # Binary classification
    prediction = 1 if prediction > 0.5 else 0
    # print(f'Text: {text}')
    # print(f'Sentiment: {prediction}')
    return prediction


# Load test data from csv file from column 'cleaned_text'
test_df = pd.read_csv('data/cleaned_Prediction_file.csv', encoding='utf-8', sep=';')

texts = test_df['cleaned_text']

# Predict the sentiment of each text
predictions = [predict_sentiment(text) for text in tqdm(
    texts, desc='Predicting sentiment', total=len(texts)
)]

# Save the predictions to a csv file
test_df['sentiment'] = predictions
test_df.to_csv('output/predictions.csv', index=False)
