import os

def load_texts_until_wordcount(folder_path, target_wordcount=200000):
    """
    Lädt rekursiv alle .txt Dateien aus folder_path (inklusive Unterordner),
    kombiniert den Text und stoppt, sobald target_wordcount Wörter erreicht sind.
    
    Args:
        folder_path (str): Pfad zum DATA-Ordner.
        target_wordcount (int): Zielwortanzahl für den kombinierten Text.
    
    Returns:
        str: Kombinierter Text mit mindestens target_wordcount Wörtern.
    """
    collected_texts = []
    total_words = 0
    
    # Alle .txt Dateien im Verzeichnisbaum finden
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    txt_files.sort()  # Sortieren für Konsistenz
    
    print(f"Gefundene Textdateien: {len(txt_files)}")
    
    for filepath in txt_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().lower()
            word_count = len(text.split())
            
            if total_words + word_count <= target_wordcount:
                collected_texts.append(text)
                total_words += word_count
                print(f"Datei {os.path.basename(filepath)} hinzugefügt, Gesamtwörter: {total_words}")
            else:
                # Nur den noch fehlenden Teil hinzufügen
                needed_words = target_wordcount - total_words
                words = text.split()
                partial_text = ' '.join(words[:needed_words])
                collected_texts.append(partial_text)
                total_words += needed_words
                print(f"Wortanzahl-Ziel erreicht mit {total_words} Wörtern nach Datei {os.path.basename(filepath)}")
                break
    
    combined_text = '\n'.join(collected_texts)
    print(f"Gesamtlänge kombinierter Text: {total_words} Wörter")
    return combined_text

if __name__ == "__main__":
    folder = r"D:\Programming_DeepLearning\EA_3_Language_Model_mit_LSTM\CodEAlltag_pXL_GERMAN-main\data"  # Pfad anpassen!
    text_for_training = load_texts_until_wordcount(folder)
    
    # Beispiel: Speichern in Datei (optional)
    with open('combined_training_text.txt', 'w', encoding='utf-8') as f:
        f.write(text_for_training)
    print("Kombinierter Trainings-Text gespeichert in combined_training_text.txt")
