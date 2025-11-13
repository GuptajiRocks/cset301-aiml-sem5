import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Attempt to import SMOTE. Note: This requires the imblearn library.
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: imblearn library not found. Please install it ('pip install imblearn') to run the SMOTE experiment.")
    exit()
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, SimpleRNN, Bidirectional, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# --- Configuration Constants ---
MAX_WORDS = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
RNN_UNITS = 128 
EPOCHS = 15
BATCH_SIZE = 32

# ----------------------------------------------------------------------
# 1. Data Preprocessing
# ----------------------------------------------------------------------

print("--- 1a) Data Loading and Initial Inspection ---")
try:
    df = pd.read_csv("nov13\\7817_1.csv", usecols=['reviews.text', 'reviews.rating'])
except FileNotFoundError:
    print("Error: '7817_1.csv' not found. Please ensure the file is accessible.")
    exit()

df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)

# 1b) Clean and preprocess the text data
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

# 1c) Encode the star ratings into three sentiment categories
print("\n--- 1c) Sentiment Encoding ---")
def encode_sentiment(rating):
    try:
        rating = float(rating)
        if rating in [1.0, 2.0]: return 0  # Negative
        elif rating == 3.0: return 1  # Neutral
        elif rating in [4.0, 5.0]: return 2  # Positive
    except (ValueError, TypeError):
        return None
df['sentiment'] = df['reviews.rating'].apply(encode_sentiment)
df.dropna(subset=['sentiment'], inplace=True)
df['sentiment'] = df['sentiment'].astype(int)

y = to_categorical(df['sentiment'], num_classes=3)
X = df['cleaned_text'].values

# 1e) Split the dataset into training and testing sets
print("\n--- 1e) Data Splitting ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['sentiment'], random_state=42)

# 1d) Convert the cleaned text into numerical sequences using TextVectorization
print("\n--- 1d) Text Vectorization ---")
vectorize_layer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)
vectorize_layer.adapt(X_train)
VOCAB_SIZE = len(vectorize_layer.get_vocabulary())

X_train_vec = vectorize_layer(X_train)
X_test_vec = vectorize_layer(X_test)


# ----------------------------------------------------------------------
# SMOTE Implementation (Experimental)
# ----------------------------------------------------------------------
print("\n" + "="*50)
print("SMOTE Resampling (Experimental)")
print("="*50)

# SMOTE requires a 2D feature matrix and 1D label vector.

# 1. Flatten the vectorized training data (Features)
X_train_flat = X_train_vec.numpy().reshape(X_train_vec.shape[0], -1)

# 2. Convert one-hot encoded labels back to integer labels (Labels)
y_train_labels = np.argmax(y_train, axis=1)

print(f"Original training data shape (flat): {X_train_flat.shape}")
print(f"Original label distribution: {pd.Series(y_train_labels).value_counts()}")

# 3. Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled_flat, y_resampled_labels = sm.fit_resample(X_train_flat, y_train_labels)

# 4. Reshape features back to 3D for the Embedding layer
X_train_resampled = X_resampled_flat.reshape(X_resampled_flat.shape[0], SEQUENCE_LENGTH)

# 5. Convert labels back to one-hot encoding
y_train_resampled = to_categorical(y_resampled_labels, num_classes=3)

print(f"Resampled training data shape (BiRNN input): {X_train_resampled.shape}")
print(f"Resampled label distribution: {pd.Series(y_resampled_labels).value_counts()}")

# ----------------------------------------------------------------------
# 2. Bidirectional SimpleRNN Model Implementation and Training (on SMOTE data)
# ----------------------------------------------------------------------
print("\n" + "="*50)
print("2. Bidirectional SimpleRNN Model Implementation and Training (on SMOTE data)")
print("="*50)

# Build the Bi-SimpleRNN Model
optimized_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    # Bidirectional SimpleRNN layer
    Bidirectional(SimpleRNN(RNN_UNITS)),
    Dropout(0.5),
    Dense(3, activation='softmax')
], name="SMOTE_BiRNN_Model")

optimized_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nSMOTE Bi-RNN Model Summary:")
optimized_model.summary()

# Train the model using SMOTE data. Note: No class weights are used here as data is balanced.
print("\nStarting SMOTE Bi-RNN Model Training...")
optimized_history = optimized_model.fit(
    X_train_resampled, 
    y_train_resampled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_vec, y_test),
    verbose=1
)
print("SMOTE Bi-RNN Model Training Complete.")


# ----------------------------------------------------------------------
# 3. Result Evaluation and Visualization
# ----------------------------------------------------------------------
print("\n" + "="*50)
print("3. Result Evaluation and Visualization")
print("="*50)

# 3b) Evaluate the model on the test data
print("\n--- SMOTE Bi-RNN Model Evaluation ---")
optimized_loss, optimized_accuracy = optimized_model.evaluate(X_test_vec, y_test, verbose=0)
print(f"Validation Accuracy: {optimized_accuracy:.4f}")
print(f"Validation Loss: {optimized_loss:.4f}")

# Predict on test data
y_pred_optimized = np.argmax(optimized_model.predict(X_test_vec, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

# 3c) Generate a confusion matrix and a classification report
print("\n--- SMOTE Bi-RNN Classification Report ---")
print(classification_report(y_true, y_pred_optimized, target_names=['Negative', 'Neutral', 'Positive']))

print("\n--- SMOTE Bi-RNN Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred_optimized))


# 3a) Plot training and validation accuracy and loss
print("\n--- 3a) Plotting Accuracy and Loss ---")

def plot_history(history, metric, title):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history[metric], label=f'Train {metric.capitalize()} (SMOTE)')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

# Plot Accuracy
plot_history(optimized_history, 'accuracy', 'SMOTE Bi-RNN Model Accuracy')
print("Saved plot: SMOTE Bi-RNN Model Accuracy")

# Plot Loss
plot_history(optimized_history, 'loss', 'SMOTE Bi-RNN Model Loss')
print("Saved plot: SMOTE Bi-RNN Model Loss")