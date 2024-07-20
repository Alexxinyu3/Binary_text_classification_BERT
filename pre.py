import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Load the Data
file_path = 'D:/Desktop/study in France/ESIGELEC-study/Intership/IPSOS/Prediction_file.xlsx'
data = pd.read_excel(file_path)

# Step 2: Overview of the Data
print("First few rows of the dataset:")
print(data.head())

print("\nSummary statistics:")
print(data.describe(include='all'))

print("\nData info:")
data.info()

# Step 3: Distribution of Columns
# Create a directory for plots if it doesn't exist
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Analyze the length distribution of Sound Bite Text_clean
data['text_length'] = data['Sound Bite Text_clean'].apply(len)

plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], kde=True)
plt.title('Distribution of Text Comment Lengths')
plt.xlabel('Text Comment Length')
plt.ylabel('Frequency')
# Save the plot to a file
text_length_plot_path = os.path.join(plot_dir, 'text_length_distribution.png')
plt.savefig(text_length_plot_path)
plt.close()

print("\nPlots have been saved to the 'plots' directory.")