import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

# --- New Import ---
# You may need to install this: pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# We no longer need class_weight
# from sklearn.utils import class_weight 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SimpleRNN, TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --- Download NLTK resources ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Configuration ---
FILE_PATH = 'nov13\\7817_1.csv'
TEXT_COLUMN = 'reviews.text'
RATING_COLUMN = 'reviews.rating'

# Model constants
VOCAB_SIZE = 10000
MAX_LEN = 150  # Fixed sequence length from prompt (1d)
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10  # Main training epochs
TUNER_EPOCHS = 4 # Epochs for hyperparameter search

# --- 1. Data Preprocessing ---

print("--- Part 1: Data Preprocessing ---")

# a) Load the dataset
try:
    df = pd.read_csv(FILE_PATH, usecols=[RATING_COLUMN, TEXT_COLUMN])
except FileNotFoundError:
    print(f"Error: File '{FILE_PATH}' not found.")
    print("Please make sure '7817_1.csv' is in the same directory as the script.")
    exit()

print(f"Total number of samples loaded: {len(df)}")
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN])
print(f"Total samples after dropping NA: {len(df)}")

# c) Encode star ratings into sentiment categories
def map_sentiment(rating):
    rating = int(rating)
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    elif rating in [4, 5]:
        return 2  # Positive

df['sentiment'] = df[RATING_COLUMN].apply(map_sentiment)

# Check for imbalance
print("\nSentiment category distribution (checking for imbalance):")
print(df['sentiment'].value_counts())

# b) Clean and preprocess the text data
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # Lowercasing
    text = re.sub(r'http\S+', '', text)  # Removal of URLs
    text = re.sub(r'\d+', '', text)  # Removal of numbers
    # Removal of punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    
    # Tokenization, Stopword Removal, and Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return ' '.join(tokens)

print("\nCleaning text data...")
df['cleaned_text'] = df[TEXT_COLUMN].apply(clean_text)
print("Text cleaning complete.")

# e) Split the dataset into training and testing sets
X = df['cleaned_text']
y = df['sentiment']

# Convert labels to categorical
y_categorical = to_categorical(y, num_classes=3)

# Stratified split to maintain class distribution
X_train, X_test, y_train_cat, y_test_cat = train_test_split(
    X, y_categorical, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_categorical
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# d) Convert cleaned text into numerical sequences
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LEN  # Fix sequence length (1d)
)

# Adapt the vectorizer to the training text
print("Adapting TextVectorization layer...")
vectorize_layer.adapt(X_train)

# Vectorize the train and test data
X_train_vec = vectorize_layer(X_train)
X_test_vec = vectorize_layer(X_test)
y_train_labels = np.argmax(y_train_cat, axis=1) # Get original labels for oversampling

# --- Handle Class Imbalance (NEW STRATEGY: Oversampling) ---
print("\nHandling class imbalance with RandomOverSampler...")
ros = RandomOverSampler(random_state=42)

# We resample the *vectorized* training data and the labels
# Note: fit_resample expects 1D labels, not categorical
X_train_res, y_train_res_labels = ros.fit_resample(X_train_vec, y_train_labels)

# Convert resampled labels back to categorical
y_train_res_cat = to_categorical(y_train_res_labels, num_classes=3)

print(f"Original training set shape: {X_train_vec.shape}")
print(f"Resampled training set shape: {X_train_res.shape}")
print("Resampled training label distribution:")
print(pd.Series(y_train_res_labels).value_counts())


# --- Hyperparameter Tuning (User Request) ---
print("\n--- Starting Hyperparameter Tuning (using KerasTuner) ---")

def build_model_tuner(hp):
    """Builds a model for KerasTuner."""
    
    # Tune the number of units in the LSTM layer
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    # Tune the dropout rate
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(units=hp_units),
        Dropout(rate=hp_dropout),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=hp_learning_rate),
        metrics=['accuracy']
    )
    return model

# --- UPDATED TUNER DATA ---
# We create a stratified validation split from our *new resampled data*
# This is much more reliable than using a small, unstratified subset
print("Creating stratified validation split for tuner...")
X_tuner_train, X_tuner_val, y_tuner_train, y_tuner_val = train_test_split(
    X_train_res, y_train_res_cat, 
    test_size=0.2,  # 20% of the resampled data for validation
    random_state=42, 
    stratify=y_train_res_cat
)

tuner = kt.Hyperband(
    build_model_tuner,
    objective='val_accuracy',
    max_epochs=TUNER_EPOCHS,
    factor=3,
    directory='kt_dir',
    project_name='sentiment_tuning'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

print("Running tuner search... (This may take a while)")
tuner.search(
    X_tuner_train, 
    y_tuner_train, 
    epochs=TUNER_EPOCHS, 
    validation_data=(X_tuner_val, y_tuner_val), # Use our new split
    callbacks=[stop_early],
    verbose=1 # Set to 1 for more details
)
print("Tuner search complete.")

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nBest Hyperparameters found:")
print(f"Units: {best_hps.get('units')}")
print(f"Dropout: {best_hps.get('dropout'):.2f}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")


# --- 2. Model Implementation ---

print("\n--- Part 2: Model Implementation ---")

# a) Build the LSTM model using best hyperparameters
def build_lstm_model(hps):
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(units=hps.get('units')),
        Dropout(rate=hps.get('dropout')),
        Dense(3, activation='softmax')
    ])
    
    # b) Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=hps.get('learning_rate')),
        metrics=['accuracy']
    )
    return model

model_lstm = build_lstm_model(best_hps)
print("\nLSTM Model Summary:")
model_lstm.summary()

# c) Train the model
print("\nTraining LSTM model on RESAMPLED data...")
history_lstm = model_lstm.fit(
    X_train_res, y_train_res_cat, # <-- Train on resampled data
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_vec, y_test_cat), # <-- Validate on original test set
    # class_weight=class_weight_dict,  # <-- REMOVED
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)


# --- 4. Comparison with Baseline Models ---

print("\n--- Part 4: Baseline Model Comparison ---")

# a) Implement a Simple RNN model
def build_rnn_model(hps):
    # Use same HPs for a fair comparison
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        SimpleRNN(units=hps.get('units')), # The only change is here
        Dropout(rate=hps.get('dropout')),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=hps.get('learning_rate')),
        metrics=['accuracy']
    )
    return model

model_rnn = build_rnn_model(best_hps)
print("\nSimpleRNN Model Summary:")
model_rnn.summary()

# Train the SimpleRNN model
print("\nTraining SimpleRNN model on RESAMPLED data...")
history_rnn = model_rnn.fit(
    X_train_res, y_train_res_cat, # <-- Train on resampled data
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_vec, y_test_cat), # <-- Validate on original test set
    # class_weight=class_weight_dict, # <-- REMOVED
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)


# --- 3. Result Evaluation and Visualization (for LSTM) ---

print("\n--- Part 3: Result Evaluation (LSTM Model) ---")

# b) Evaluate the model on the test data
print("\nEvaluating LSTM Model on Test Data:")
val_loss, val_accuracy = model_lstm.evaluate(X_test_vec, y_test_cat)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# c) Generate a confusion matrix and a classification report
y_pred_probs = model_lstm.predict(X_test_vec)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test_cat, axis=1)

# Classification Report
print("\nClassification Report (LSTM):")
target_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print(classification_report(y_test_labels, y_pred, target_names=target_names))

# Confusion Matrix
print("Confusion Matrix (LSTM):")
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('LSTM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# --- 4b. Comparison Visualization ---

print("\n--- Part 4b: Model Comparison Plots ---")

# a) Plot training and validation accuracy across epochs
plt.figure(figsize=(14, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='LSTM Train Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Val Accuracy')
plt.plot(history_rnn.history['accuracy'], label='SimpleRNN Train Accuracy', linestyle='--')
plt.plot(history_rnn.history['val_accuracy'], label='SimpleRNN Val Accuracy', linestyle='--')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['loss'], label='LSTM Train Loss')
plt.plot(history_lstm.history['val_loss'], label='LSTM Val Loss')
plt.plot(history_rnn.history['loss'], label='SimpleRNN Train Loss', linestyle='--')
plt.plot(history_rnn.history['val_loss'], label='SimpleRNN Val Loss', linestyle='--')
plt.title('Model Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

