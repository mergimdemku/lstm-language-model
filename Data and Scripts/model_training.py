import numpy as np
import tensorflow as tf

SEQ_LENGTH = 5

# Trainingsdaten laden
data = np.load('train_data.npz')
X = data['X']
y = data['y']

vocab_size = y.shape[1]

# Modell erstellen
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=SEQ_LENGTH),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
model.fit(X, y, epochs=10, batch_size=32)

# Modell speichern (TensorFlow SavedModel Format)
model.save('saved_lstm_model')
