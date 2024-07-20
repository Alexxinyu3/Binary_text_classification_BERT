# ---------Frequency-------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv('data/predictions.csv')

# Check the data
print(df.head())

# Frequency mapping
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Frequency Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['0', '1'])
plt.savefig('plots/sentiment_frequency_distribution.png')
plt.close()


# ---------Word Clouds-----------------------

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Read CSV file
df = pd.read_csv('data/predictions.csv')

# Separate data with sentiment columns 0 and 1
text_sentiment_0 = ' '.join(df[df['sentiment'] == 0]['cleaned_text'])
text_sentiment_1 = ' '.join(df[df['sentiment'] == 1]['cleaned_text'])

# Create a word cloud map
wordcloud_0 = WordCloud(width=800, height=400, background_color='white').generate(text_sentiment_0)
wordcloud_1 = WordCloud(width=800, height=400, background_color='white').generate(text_sentiment_1)

# Mapping the word cloud
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_0, interpolation='bilinear')
plt.axis('off')
plt.title('Sentiment 0')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_1, interpolation='bilinear')
plt.axis('off')
plt.title('Sentiment 1')

# save
wordcloud_0.to_file('plots/wordcloud_sentiment_0.png')
wordcloud_1.to_file('plots/wordcloud_sentiment_1.png')