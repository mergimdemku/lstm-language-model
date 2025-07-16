let model;
const SEQ_LENGTH = 5;
let tokenizerLoaded = false;
let inputSequence = [];
let wordIndex = {};
let indexWord = {};

// Tokenizer laden (jetzt tokenizer_word_index.json)
async function loadTokenizer() {
  try {
    const response = await fetch('tokenizer_word_index.json');
    if (!response.ok) throw new Error('Tokenizer nicht gefunden');
    wordIndex = await response.json();

    if (!wordIndex) throw new Error('word_index im Tokenizer fehlt');

    // indexWord umkehren: ID → Wort
    indexWord = {};
    for (const [word, index] of Object.entries(wordIndex)) {
      indexWord[index] = word;
    }
    tokenizerLoaded = true;
  } catch (err) {
    alert('Fehler beim Laden des Tokenizers: ' + err.message);
    console.error(err);
  }
}

// Nächstes Wort vorhersagen
async function predictNextWord() {
  if (!tokenizerLoaded) {
    alert('Tokenizer noch nicht geladen!');
    return;
  }
  if (inputSequence.length < SEQ_LENGTH) {
    alert(`Bitte mindestens ${SEQ_LENGTH} Wörter eingeben.`);
    return;
  }
  const inputTensor = tf.tensor2d([inputSequence.slice(-SEQ_LENGTH)], [1, SEQ_LENGTH]);
  const prediction = model.predict(inputTensor);
  const predictedIdTensor = prediction.argMax(1);
  const predictedId = (await predictedIdTensor.data())[0];
  predictedIdTensor.dispose();
  prediction.dispose();
  inputTensor.dispose();

  const predictedWord = indexWord[predictedId] || '<unk>';

  inputSequence.push(predictedId);
  updateOutput(predictedWord);
}

// Automatische Vorhersage von 10 Wörtern
async function predictAuto() {
  for (let i = 0; i < 10; i++) {
    await predictNextWord();
  }
}

// Ausgabe aktualisieren
function updateOutput(word) {
  const outputDiv = document.getElementById('output');
  outputDiv.textContent += ' ' + word;
}

// Reset Funktion
function reset() {
  inputSequence = [];
  document.getElementById('output').textContent = '';
  document.getElementById('inputText').value = '';
}

// Eingabetext verarbeiten
function processInputText(text) {
  if (!tokenizerLoaded) return;
  const words = text.trim().toLowerCase().split(/\s+/);
  inputSequence = words.map(w => wordIndex[w] || 0);
  document.getElementById('output').textContent = text;
}

// Init
window.onload = async () => {
  await loadTokenizer();

  try {
    model = await tf.loadLayersModel('web_model/model.json');
  } catch (err) {
    alert('Fehler beim Laden des Modells: ' + err.message);
    console.error(err);
  }

  document.getElementById('predictBtn').onclick = predictNextWord;
  document.getElementById('autoPredictBtn').onclick = predictAuto;
  document.getElementById('resetBtn').onclick = reset;

  document.getElementById('inputText').addEventListener('input', (e) => {
    processInputText(e.target.value);
  });
};
