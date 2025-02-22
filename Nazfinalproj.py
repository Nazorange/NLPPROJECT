import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Load datasets
train_file_path = 'D:/NLP/FINAL PROJ/FinalProjectData/kagglemoviereviews/corpus/train.tsv'
test_file_path = 'D:/NLP/FINAL PROJ/FinalProjectData/kagglemoviereviews/corpus/test.tsv'
lexicon_file_path = 'D:/NLP/FINAL PROJ/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff'

train_data = pd.read_csv(train_file_path, sep='\t')
test_data = pd.read_csv(test_file_path, sep='\t')

# Preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text_with_negation(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Negation handling
    for i in range(len(tokens) - 1):
        if tokens[i] in ['not', 'no', "don't", "won't"]:
            tokens[i] = tokens[i] + '_' + tokens[i + 1]
            tokens[i + 1] = ''
    tokens = [word for word in tokens if word]
    return ' '.join(tokens)

# Remove duplicates in train_data
train_data = train_data.drop_duplicates(subset=['Phrase']).reset_index(drop=True)

# Align labels with deduplicated data
y_train = train_data['Sentiment'].reset_index(drop=True)

# Debug Shape
print(f"Shape of train_data after deduplication: {train_data.shape}")
print(f"Shape of y_train after alignment: {y_train.shape}")

train_data['ProcessedText'] = train_data['Phrase'].apply(preprocess_text_with_negation)
test_data['ProcessedText'] = test_data['Phrase'].apply(preprocess_text_with_negation)

# TF-IDF Features
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['ProcessedText']).toarray()
X_test_tfidf = tfidf_vectorizer.transform(test_data['ProcessedText']).toarray()

# Debug Shape
print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape of y_train: {y_train.shape}")

# Sentiment Lexicon Features
def load_subjectivity_lexicon(path):
    pos_words, neg_words = set(), set()
    with open(path, 'r') as file:
        for line in file:
            if "priorpolarity=positive" in line:
                pos_words.add(line.split()[2].split('=')[1])
            elif "priorpolarity=negative" in line:
                neg_words.add(line.split()[2].split('=')[1])
    return pos_words, neg_words

pos_words, neg_words = load_subjectivity_lexicon(lexicon_file_path)

def sentiment_features(text):
    tokens = text.split()
    pos_count = sum(1 for word in tokens if word in pos_words)
    neg_count = sum(1 for word in tokens if word in neg_words)
    return [pos_count, neg_count]

train_data['LexiconFeatures'] = train_data['ProcessedText'].apply(sentiment_features)
test_data['LexiconFeatures'] = test_data['ProcessedText'].apply(sentiment_features)

# POS Tagging Features
def pos_features(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    return [noun_count, verb_count, adj_count]

train_data['POSFeatures'] = train_data['ProcessedText'].apply(pos_features)
test_data['POSFeatures'] = test_data['ProcessedText'].apply(pos_features)

# Combine Features
X_train_combined = pd.concat([
    pd.DataFrame(X_train_tfidf),
    pd.DataFrame(train_data['LexiconFeatures'].tolist()),
    pd.DataFrame(train_data['POSFeatures'].tolist())
], axis=1).values

X_test_combined = pd.concat([
    pd.DataFrame(X_test_tfidf),
    pd.DataFrame(test_data['LexiconFeatures'].tolist()),
    pd.DataFrame(test_data['POSFeatures'].tolist())
], axis=1).values

# Scale Features
scaler = MinMaxScaler()
X_train_combined = scaler.fit_transform(X_train_combined)
X_test_combined = scaler.transform(X_test_combined)

# Compute the number of samples for each class (increase by 50% of the minority class)
print(f"Original class distribution: {Counter(y_train)}")
sampling_strategy = {
    label: max(count + int(0.5 * count), count)
    for label, count in Counter(y_train).items()
}
# Initialize SMOTE with the computed strategy
smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train)

# Debug Shape
print(f"Shape of X_train_combined after SMOTE: {X_train_smote.shape}")
print(f"Shape of y_train after SMOTE: {y_train_smote.shape}")
print(f"Class distribution after SMOTE: {Counter(y_train_smote)}")

# Evaluation
results_log = []

def evaluate_model(model, model_name, X_train, y_train, X_test):
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
    results_log.append({
        "Model": model_name,
        "Cross-Validation F1 Scores": scores.tolist(),
        "Mean F1 Score": scores.mean()
    })
    print(f"{model_name} - Cross-Validation F1 Scores: {scores}, Mean F1 Score: {scores.mean()}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Evaluate Baseline TF-IDF Only (Original Data)
print("\nEvaluating Baseline TF-IDF Only...")
nb_model = MultinomialNB()
y_pred_nb_tfidf = evaluate_model(nb_model, "Naive Bayes (TF-IDF Only)", X_train_tfidf, y_train, X_test_tfidf)

# Evaluate Combined Features with Logistic Regression (SMOTE Data)
print("\nEvaluating Logistic Regression with Combined Features...")
log_reg_model = LogisticRegression(max_iter=1000, solver='saga')
y_pred_lr_combined = evaluate_model(log_reg_model, "Logistic Regression (Combined Features)", X_train_smote, y_train_smote, X_test_combined)

# RandomizedSearchCV for Random Forest
print("\nEvaluating Random Forest with Randomized Search...")
param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
rf_tuned = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=10, cv=3, scoring='f1_weighted', n_jobs=-1, random_state=42)

# Use a Smaller Subset for Randomized Search
sample_size = 50000
X_train_sample = X_train_smote[:sample_size]
y_train_sample = y_train_smote[:sample_size]

rf_tuned.fit(X_train_sample, y_train_sample)
print(f"Best Parameters: {rf_tuned.best_params_}")
y_pred_rf = rf_tuned.predict(X_test_combined)

# Log all results
results_df = pd.DataFrame(results_log)
results_df.to_csv("D:/NLP/FINAL PROJ/model_results_log.csv", index=False)
print("All results logged in model_results_log.csv")
