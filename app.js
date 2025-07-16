let model;
const SEQ_LENGTH = 5;
let tokenizer = null;
let inputSequence = [];
let wordIndex = {};
let indexWord = {};

// Hilfsfunktion zum Laden des Tokenizers (tokenizer.json)
async function loadTokenizer() {
  const response = await fetch('tokenizer.json');
  const tokenizerJson = await response.json();

  wordIndex = tokenizerJson['word_index'];
  // IndexWord aufbauen (id → Wort)
  indexWord = {};
  for (const [word, index] of Object.entries(wordIndex)) {
    indexWord[index] = word;
  }
  tokenizer = true; // Dummy, nur um zu zeigen, dass geladen ist
}

// Vorhersage eines nächsten Worts
async function predictNextWord() {
  if (!tokenizer) {
    alert('Tokenizer noch nicht geladen!');
    return;
  }
  if (inputSequence.length < SEQ_LENGTH) {
    alert(`Bitte mindestens ${SEQ_LENGTH} Wörter eingeben.`);
    return;
  }

  // Letzte SEQ_LENGTH IDs als Tensor
  const inputTensor = tf.tensor2d([inputSequence.slice(-SEQ_LENGTH)], [1, SEQ_LENGTH]);

  // Modell vorhersagen lassen
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

// Mehrere Wörter automatisch vorhersagen (bis 10)
async function predictAuto() {
  for (let i = 0; i < 10; i++) {
    await predictNextWord();
  }
}

// Ausgabe im Div aktualisieren
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

// Eingabetext in Wort-IDs umwandeln
function processInputText(text) {
  if (!tokenizer) return;

  const words = text.trim().toLowerCase().split(/\s+/);
  inputSequence = words.map(w => wordIndex[w] || 0);
  const outputDiv = document.getElementById('output');
  outputDiv.textContent = text;
}

// Initialisierung beim Laden der Seite
window.onload = async () => {
  await loadTokenizer();

  // Lade das konvertierte Modell aus dem Ordner web_model
  model = await tf.loadLayersModel('web_model/model.json');

  document.getElementById('predictBtn').onclick = predictNextWord;
  document.getElementById('autoPredictBtn').onclick = predictAuto;
  document.getElementById('resetBtn').onclick = reset;

  document.getElementById('inputText').addEventListener('input', (e) => {
    processInputText(e.target.value);
  });
};
