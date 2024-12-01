import pandas as pd
from sklearn.utils import shuffle
from collections import Counter

# Wczytaj dane
with open('positive.txt', 'r', encoding='utf-8') as file:
 positive_data = file.readlines()
with open('negative.txt', 'r', encoding='utf-8') as file:
 negative_data = file.readlines()
# Przydziel klasy (0 - positive, 1 - negative)
positive_df = pd.DataFrame({'text': positive_data, 'class': 0})
negative_df = pd.DataFrame({'text': negative_data, 'class': 1})
# Połącz ramki danych
df = pd.concat([positive_df, negative_df], ignore_index=True)
# Przemieszaj zbiór danych
df = shuffle(df)
print(df.head())

all_words = " ".join(df['text']).lower().split()
bag_of_words = Counter(all_words)

print("Najczęściej występujące słowa w zbiorze: ")
print(bag_of_words.most_common(10))

negative_words = " ".join(df[df['class'] == 1]['text']).lower().split()
positive_words = " ".join(df[df['class'] == 0]['text']).lower().split()

negative_bow = Counter(negative_words)
positive_bow = Counter(positive_words)

print("\nNajczęściej występujące słowa w recenzjach negatywnych:")
print(negative_bow.most_common(10))

print("\nNajczęściej występujące słowa w recenzjach pozytywnych:")
print(positive_bow.most_common(10))
 
