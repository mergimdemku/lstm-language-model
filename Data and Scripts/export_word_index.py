import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Lade den gespeicherten Tokenizer (z.B. tokenizer.json)
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()

tokenizer = tokenizer_from_json(tokenizer_json)

# Extrahiere word_index
word_index = tokenizer.word_index

# Schreibe word_index in eigene JSON-Datei
with open('tokenizer_word_index.json', 'w', encoding='utf-8') as f:
    json.dump(word_index, f, ensure_ascii=False, indent=2)

print("tokenizer_word_index.json wurde erfolgreich erstellt.")
