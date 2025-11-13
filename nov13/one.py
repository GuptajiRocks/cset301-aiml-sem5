import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, SimpleRNN, LSTM, Bidirectional, Dropout, Dense, GlobalAveragePooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

MAX_WORDS = 10000  
SEQUENCE_LENGTH = 100  
EMBEDDING_DIM = 128
RNN_UNITS = 128  
EPOCHS = 15  
BATCH_SIZE = 32


df = pd.read_csv("nov13\\7817_1.csv", usecols=['reviews.text', 'reviews.rating'])


print(f"Total number of samples: {len(df)}")

df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)
print(f"Total samples after dropping NaNs: {len(df)}")

print("\nMissing values per column after dropna:")
print(df.isnull().sum())

print("\n--- 1b) Text Cleaning and Preprocessing ---")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z\s]', '', text) 

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

df['cleaned_text'] = df['reviews.text'].apply(clean_text)
print("Text cleaning (Lowercasing, removal of URLs, numbers, punctuation) and preprocessing (Stopword removal, Lemmatization) complete.")

print("\n--- 1c) Sentiment Encoding ---")
def encode_sentiment(rating):

    try:
        rating = float(rating)
        if rating in [1.0, 2.0]:
            return 0  
        elif rating == 3.0:
            return 1  
        elif rating in [4.0, 5.0]:
            return 2  
    except (ValueError, TypeError):
        return None 

df['sentiment'] = df['reviews.rating'].apply(encode_sentiment)

df.dropna(subset=['sentiment'], inplace=True)
df['sentiment'] = df['sentiment'].astype(int)

y = to_categorical(df['sentiment'], num_classes=3)
X = df['cleaned_text'].values
print(f"Sentiment distribution (0=Negative, 1=Neutral, 2=Positive):\n{df['sentiment'].value_counts()}")
print(f"Data shape after encoding: X={X.shape}, y={y.shape}")

print("\n--- 1e) Data Splitting ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['sentiment'], random_state=42)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

train_labels_int = np.argmax(y_train, axis=1)

class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_int),
    y=train_labels_int
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

print("\n--- Class Imbalance Handling (Class Weighting) ---")
print("Classes are imbalanced, using class weights to balance loss:")
print(f"Class Weights (0:Negative, 1:Neutral, 2:Positive): {class_weights}")
print("These weights will be applied during model training to penalize minority misclassifications more heavily.")

print("\n--- 1d) Text Vectorization ---")
vectorize_layer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

vectorize_layer.adapt(X_train)
VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
print(f"Vocabulary size after adaptation: {VOCAB_SIZE}")

X_train_vec = vectorize_layer(X_train)
X_test_vec = vectorize_layer(X_test)

print("\n" + "="*50)
print("2. Optimized Bidirectional LSTM Model Implementation and Training")
print("="*50)

optimized_model = Sequential([

    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),

    Bidirectional(LSTM(RNN_UNITS)),
    Dropout(0.5),
    Dense(3, activation='softmax') 
], name="Optimized_LSTM_Model")

optimized_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nOptimized LSTM Model Summary:")
optimized_model.summary()

print("\nStarting Optimized LSTM Model Training (with Class Weights)...")
optimized_history = optimized_model.fit(
    X_train_vec, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_vec, y_test),
    class_weight=class_weights,
    verbose=1
)
print("Optimized LSTM Model Training Complete.")

print("\n" + "="*50)
print("2. SimpleRNN Model Implementation and Training (for comparison)")
print("="*50)
rnn_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    SimpleRNN(RNN_UNITS // 2), 
    Dropout(0.5),
    Dense(3, activation='softmax')
], name="SimpleRNN_Model")

rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("\nSimpleRNN Model Summary:")
rnn_model.summary()
print("\nStarting SimpleRNN Model Training (with Class Weights)...")
rnn_history = rnn_model.fit(
    X_train_vec, 
    y_train,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    validation_data=(X_test_vec, y_test),
    class_weight=class_weights, 
    verbose=1
)
print("SimpleRNN Model Training Complete.")

print("\n" + "="*50)
print("4a) Baseline ANN Model Implementation and Training")
print("="*50)

ann_model = Sequential([

    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),

    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax') 
], name="BaselineANN_Model")

ann_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nANN Model Summary:")
ann_model.summary()

print("\nStarting ANN Model Training (with Class Weights)...")
ann_history = ann_model.fit(
    X_train_vec, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_vec, y_test),
    class_weight=class_weights, 
    verbose=1
)
print("ANN Model Training Complete.")

print("\n" + "="*50)
print("3. Result Evaluation and Visualization (Optimized LSTM vs SimpleRNN vs ANN)")
print("="*50)

print("\n--- Optimized Bi-LSTM Model Evaluation ---")
optimized_loss, optimized_accuracy = optimized_model.evaluate(X_test_vec, y_test, verbose=0)
print(f"Validation Accuracy: {optimized_accuracy:.4f}")
print(f"Validation Loss: {optimized_loss:.4f}")

y_pred_optimized = np.argmax(optimized_model.predict(X_test_vec, verbose=0), axis=1)

y_true = np.argmax(y_test, axis=1)
print("\n--- Optimized Bi-LSTM Classification Report ---")
print(classification_report(y_true, y_pred_optimized, target_names=['Negative', 'Neutral', 'Positive']))

print("\n--- Optimized Bi-LSTM Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred_optimized))

print("\n--- SimpleRNN Model Evaluation ---")
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test_vec, y_test, verbose=0)
print(f"Validation Accuracy: {rnn_accuracy:.4f}")
print(f"Validation Loss: {rnn_loss:.4f}")

print("\n--- ANN Model Evaluation ---")
ann_loss, ann_accuracy = ann_model.evaluate(X_test_vec, y_test, verbose=0)
print(f"Validation Accuracy: {ann_accuracy:.4f}")
print(f"Validation Loss: {ann_loss:.4f}")

print("\n--- 3a) Plotting Accuracy and Loss ---")

def plot_history_comparison(histories, metric, title):
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    styles = ['-', '--', ':']

    plt.plot(histories[0].history[metric], color=colors[0], linestyle=styles[0], label=f'{histories[0].model.name} Train {metric.capitalize()}')
    plt.plot(histories[0].history[f'val_{metric}'], color=colors[0], linestyle=styles[1], label=f'{histories[0].model.name} Validation {metric.capitalize()}')

    plt.plot(histories[1].history[metric], color=colors[1], linestyle=styles[0], label=f'{histories[1].model.name} Train {metric.capitalize()}')
    plt.plot(histories[1].history[f'val_{metric}'], color=colors[1], linestyle=styles[1], label=f'{histories[1].model.name} Validation {metric.capitalize()}')

    plt.plot(histories[2].history[metric], color=colors[2], linestyle=styles[0], label=f'{histories[2].model.name} Train {metric.capitalize()}')
    plt.plot(histories[2].history[f'val_{metric}'], color=colors[2], linestyle=styles[1], label=f'{histories[2].model.name} Validation {metric.capitalize()}')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

all_histories = [optimized_history, rnn_history, ann_history]

plot_history_comparison(all_histories, 'accuracy', 'Model Accuracy Comparison (LSTM vs SimpleRNN vs ANN)')

plot_history_comparison(all_histories, 'loss', 'Model Loss Comparison (LSTM vs SimpleRNN vs ANN)')

print("\n" + "="*50)
print("4c) Discussion: Model Performance Comparison")
print("="*50)

print(f"\nFinal Optimized Bi-LSTM Validation Accuracy: {optimized_accuracy:.4f}")
print(f"Final SimpleRNN Validation Accuracy: {rnn_accuracy:.4f}")
print(f"Final Baseline ANN Validation Accuracy: {ann_accuracy:.4f}")

print("\nAnalysis of Optimizations:")
print("1. **Addressing Imbalance (Class Weighting):** We applied class weighting, which assigns a higher penalty during training when the model misclassifies a sample from the minority class (Negative or Neutral). This compels the model to pay more attention to the underrepresented classes, which should improve metrics like F1-score and recall for those classes, even if overall accuracy decreases slightly.")
print("2. **SimpleRNN vs. Bi-LSTM:** The Bi-LSTM is better at capturing long-term dependencies in the text than the SimpleRNN, which is crucial for full reviews.")
print(f"   - Expected Outcome: The Bi-LSTM model ({optimized_accuracy:.4f}) is expected to maintain its performance edge over the SimpleRNN ({rnn_accuracy:.4f}).")
print("3. **RNN/LSTM vs. ANN Baseline:** The recurrent models still leverage sequence information lost by the ANN.")
print(f"   - Expected Outcome: Both recurrent models are expected to outperform the ANN baseline ({ann_accuracy:.4f}) in most cases.")

print("\nFurther potential optimizations to consider if performance is still lacking:")
print(" - **Hyperparameter Tuning:** Experiment with different `RNN_UNITS`, `BATCH_SIZE`, and `Dropout` rates.")
print(" - **Pre-trained Embeddings:** Replace the basic `Embedding` layer with pre-trained word embeddings (like Word2Vec or GloVe) to give the model a better semantic starting point.")
print(" - **Architecture:** Consider stacking multiple Bi-LSTM layers or replacing the LSTM with a GRU layer.")