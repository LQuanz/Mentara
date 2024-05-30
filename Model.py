import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical


data = pd.read_csv("NewDataset.csv")
def remove_punctuation(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\(|\)|,|-|/', '', text)
    return text

data['Question'] = data['Question'].apply(remove_punctuation)
data['Answer'] = data['Answer'].apply(remove_punctuation)

pairs = [(question, answer) for question, answer in zip(data['Question'], data['Answer'])]
input_docs, target_docs = zip(*pairs)
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_docs + target_docs)

num_encoder_tokens = len(tokenizer.word_index) + 1
num_decoder_tokens = num_encoder_tokens

num_samples = 100
encoder_input_data = tokenizer.texts_to_matrix(input_docs, mode='binary').reshape((len(input_docs), -1, 1))
encoder_input_data = encoder_input_data[:num_samples]
decoder_input_data = tokenizer.texts_to_matrix(target_docs, mode='binary').reshape((len(input_docs), -1, 1))
decoder_input_data = decoder_input_data[:num_samples]
decoder_target_data = tokenizer.texts_to_matrix(target_docs, mode='binary').reshape((len(input_docs), -1, 1))
decoder_target_data = decoder_target_data[:num_samples]
decoder_target_data = to_categorical(decoder_target_data, num_classes=num_decoder_tokens)

dimensionality = 256
batch_size = 10
epochs = 50

encoder_inputs = Input(shape=(None, 1))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Define Decoder
decoder_inputs = Input(shape=(None, 1))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)