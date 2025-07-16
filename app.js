let model;
const SEQ_LENGTH = 5;
let tokenizer = null;
let inputSequence = [];
let wordIndex = {};
let indexWord = {};

async function loadTokenizer() {
  try {
    const response = await fetch('tokenizer.json');
    if (!response.ok) throw new Error('Tokenizer nicht gefunden');
    const tokenizerJson = await response.json();

    // Beispiel: Suche word_index im config-Objekt
    const wordIndex = tokenizerJson.config && tokenizerJson.config.word_index;
    if (!wordIndex) throw new Error('word_index im Tokenizer fehlt');

    indexWord = {};
    for (const [word, index] of Object.entries(wordIndex)) {
      indexWord[index] = word;
    }
    tokenizer = true;
  } catch (err) {
    alert('Fehler beim Laden des Tokenizers: ' + err.message);
    console.error(err);
  }
}



async function predictNextWord() {
  if (!tokenizer) {
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

async function predictAuto() {
  for (let i = 0; i < 10; i++) {
    await predictNextWord();
  }
}

function updateOutput(word) {
  const outputDiv = document.getElementById('output');
  outputDiv.textContent += ' ' + word;
}

function reset() {
  inputSequence = [];
  document.getElementById('output').textContent = '';
  document.getElementById('inputText').value = '';
}

function processInputText(text) {
  if (!tokenizer) return;
  const words = text.trim().toLowerCase().split(/\s+/);
  inputSequence = words.map(w => wordIndex[w] || 0);
  document.getElementById('output').textContent = text;
}

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
