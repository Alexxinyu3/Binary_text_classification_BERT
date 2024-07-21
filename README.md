# Sentiment Classification from scratch.
# 1. INTRODUCTION:

This research aims to utilize a deep learning algorithm to construct a text classification model that can distinguish between negative and positive sentiment data. Specifically, we will train a binary text classification model to achieve this goal.

# 2. Exploratory Data Analysis

The EDA process is crucial to understand the dataset (Prediction_file) and explore the distributions of the columns included.

Run 'preprocessing'

output:Output some relevant data distribution characteristics of the 'prediction_file'，and save the plot of distribution of text lengths to 'plots/text_length_distribution.png'

# 3. Data clean

Run'Data_clean.py'

Based on 'data_for_model',which manual labeling of 100 positive emotions (1) and 100 negative emotions (0) using the NLTK natural language toolkit, the following cleaning steps were applied:

· Remove HTML tags: Use BeautifulSoup to remove HTML tags from text.

· Remove special characters and punctuation: Use regular expressions to remove non-alphanumeric characters.

· Convert to lowercase: Normalize text by converting all characters to lowercase.

· Remove stopwords: Exclude common stopwords (such as “the”, “and”, etc.).

· Lemmatize words: Reduce words to their base form (root) using a lemmatizer.

· Remove extra spaces: Eliminate unnecessary whitespace.


output:The output is saved to'data/cleaned_data_for_model.xlsx'

# 4.Model building and evaluation

Run'Model_Building_BERT'

A BERT model is trained and evaluated for a text categorization using the following steps:

·  Load the Data: Import the dataset into the environment.

·  Split the Data: Divide the data into training and evaluation sets.

·  Tokenize and Encode: Use the BERT tokenizer to preprocess the text data.

·  Create Datasets and DataLoaders: Construct PyTorch Dataset and DataLoader objects for batching.

·  Fine-tune BERT: Adapt a pre-trained BERT model for the text classification task.

·  Train the Model: Execute the training loop.

·  Evaluate the Model: Assess model performance on the evaluation set.

·  Plot Losses and Accuracies: Visualize training progress and results.


output: 'precision'	'recall'	'f1-score'	'support' 'training loss curve' 'evaluating loss curve' 'Accurancy',and save the optimal weights of the model to the'output/weights/best.path'

# 5.Project BERT model on Prediction_file

Save the 'prediction_file.xlsx' after cleaning it to 'data/cleaned_prediction_file.xlsx'

Run'inference.py'

output:Save the prediction results to 'data/predictions.csv',and draw 'wordcloud_sentiment_0.png' and 'wordcloud_sentiment_1.png' and save them to 'plots'








