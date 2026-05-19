import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/clean_data.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

if 'clean_text' in df.columns:
    text_col = 'clean_text'
elif 'text' in df.columns:
    text_col = 'text'
else:
    raise ValueError("No valid text column found!")

df = df.dropna(subset=[text_col, 'label'])

df['label'].value_counts().sort_index().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Label (0=Human, 1=AI)")
plt.ylabel("Count")
plt.savefig("results/charts/class_distribution.png")
plt.close()

df['text_length'] = df[text_col].apply(lambda x: len(str(x).split()))
df.groupby('label')['text_length'].mean().plot(kind='bar')
plt.title("Average Text Length Per Class")
plt.xlabel("Label (0=Human, 1=AI)")
plt.ylabel("Average Word Count")
plt.savefig("results/charts/text_length_comparison.png")
plt.close()

print("EDA charts saved successfully.")