from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "Patient feels good",
    "Patient has severe chest pain",
    "Breathing difficulty",
    "Normal health condition"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)