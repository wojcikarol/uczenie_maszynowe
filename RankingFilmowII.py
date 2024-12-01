import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Wczytaj dane
with open('positive.txt', 'r', encoding='utf-8') as file:
    positive_data = file.readlines()

with open('negative.txt', 'r', encoding='utf-8') as file:
    negative_data = file.readlines()

# Przydziel klasy (0 - positive, 1 - negative)
positive_df = pd.DataFrame({'text': positive_data, 'class': 0})
negative_df = pd.DataFrame({'text': negative_data, 'class': 1})

# Połącz dane w jeden zbiór
df = pd.concat([positive_df, negative_df], ignore_index=True)

# Przemieszaj zbiór danych
df = shuffle(df)

# Podziel dane na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['class'], test_size=0.2, random_state=42
)

# Lista wektoryzatorów (różne kombinacje CountVectorizer)
vectorizers = [
    ('Default Vectorizer', CountVectorizer()),
    ('No Short Words (min 3)', CountVectorizer(token_pattern=r'\b\w{3,}\b')),
    ('Min DF 2', CountVectorizer(min_df=2)),
    ('Max Features 1000', CountVectorizer(max_features=100))
]

# Lista klasyfikatorów
classifiers = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC())
]

# Testowanie kombinacji wektoryzatorów i klasyfikatorów
results = []

for vectorizer_name, vectorizer in vectorizers:
    for classifier_name, classifier in classifiers:
        # Utwórz pipeline z wektoryzatorem i klasyfikatorem
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        # Cross-walidacja na zbiorze uczącym
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        # Trenuj model na pełnym zbiorze uczącym
        pipeline.fit(X_train, y_train)
        
        # Przewiduj na danych testowych
        y_pred = pipeline.predict(X_test)
        
        # Oceniaj wyniki
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Dodaj wyniki do listy
        results.append({
            'Vectorizer': vectorizer_name,
            'Classifier': classifier_name,
            'Mean Accuracy': cv_scores.mean(),
            'Cross-Validation Scores': cv_scores,
            'Test Accuracy': accuracy,
            'Classification Report': report
        })

# Wyświetlanie wyników
for result in results:
    print(f"Vectorizer: {result['Vectorizer']}")
    print(f"Classifier: {result['Classifier']}")
    print(f"Cross-Validation Scores: {result['Cross-Validation Scores']}")
    print(f"Mean CV Accuracy: {result['Mean Accuracy']:.4f}")
    print(f"Test Accuracy: {result['Test Accuracy']:.4f}")
    print("Classification Report:")
    print(result['Classification Report'])
    print("=" * 50)
