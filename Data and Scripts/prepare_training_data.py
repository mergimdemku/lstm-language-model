import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

SEQ_LENGTH = 5  # k = 5 Wörter Eingabe

def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text

def prepare_sequences(text, seq_length=SEQ_LENGTH):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    
    total_words = len(tokenizer.word_index) + 1
    print(f"Vokabulargröße: {total_words}")

    token_list = tokenizer.texts_to_sequences([text])[0]

    input_sequences = []
    for i in range(seq_length, len(token_list)):
        seq = token_list[i-seq_length:i+1]  # Eingabe + Zielwort
        input_sequences.append(seq)
    
    input_sequences = np.array(input_sequences)
    X = input_sequences[:, :-1]  # Eingabe
    y = input_sequences[:, -1]   # Zielwort
    y = to_categorical(y, num_classes=total_words)  # One-Hot Kodierung
    
    print(f"Anzahl Trainingsbeispiele: {X.shape[0]}")
    return X, y, tokenizer

def save_data(X, y, tokenizer):
    np.savez_compressed('train_data.npz', X=X, y=y)
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(tokenizer.to_json())
    print("Trainingsdaten in 'train_data.npz' gespeichert")
    print("Tokenizer in 'tokenizer.json' gespeichert")

if __name__ == "__main__":
    filename = 'combined_training_text.txt'  # Deine kombinierte Datei
    text = load_text(filename)
    X, y, tokenizer = prepare_sequences(text)
    save_data(X, y, tokenizer)
