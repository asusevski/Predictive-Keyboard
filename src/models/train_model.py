import glob
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import LSTM
from keras.layers import Dense
import pickle


def generate_corpus(path: str) -> str:
    """
    Generates corpus to be used as dataset from the books downloaded from Project Gutenberg.

    Arguments:
        path -- the path to the directory that contains the books

    Returns: 
        corpus -- the text contained in all the books stored as a string
    """
    books = []
    for book in glob.glob(path + '/' + '*.txt'):
        books.append(book)

    text = ""
    for book in books:
        text += open(book, 'r', encoding='utf-8').read().lower()

    return text


def generate_labelled_data(corpus: str, chunk_length: int) -> tuple[list[str], list[str]]:
    """
    Separates the corpus into chunks and labels where chunks are strings of text from the 
    corpus and labels are the character that follows any given chunk

    Example:
        If corpus was the string "Hello, my name is Inigo Montoya, you killed my father, 
        prepare to die", and we wanted chunks of length 5, the first chunk would be 
        "Hello" and the label would be the character proceeding it, namely ","

    Arguments:
        corpus -- the text used to train the LSTM
        chunk_length -- the number of characters in each training example

    Returns:
        chunks -- list of strings of text that the model will be fed as input
        labels -- list of characters proceeding the text in the given chunk

    """
    chunks = []
    labels = []
    for i in range(0, len(corpus) - chunk_length, chunk_length):
        chunks.append(corpus[i: i + chunk_length])
        labels.append(corpus[i + chunk_length])
    return chunks, labels


def generate_one_hot(corpus: str, chunk_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes corpus and returns one-hot representations of chunks and labels

    Arguments:
        corpus -- body of text to transform into one-hot vectors used to train LSTM

    Returns:
        X -- chunks of text from corpus transformed into one-hot vectors
        y -- labels of text from corpus transformed into one-hot vectors
    """
    chars = sorted(list(set(corpus)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    chunks, labels = generate_labelled_data(corpus, chunk_length)
    X = np.zeros((len(chunks), chunk_length, len(chars)))
    y = np.zeros((len(chunks), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(chunks):
        for j, char in enumerate(sentence):
            X[i, j, char_to_ix[char]] = 1
        y[i, char_to_ix[labels[i]]] = 1
    return X, y


def train_LTSM_model(model_name="lstm_model", corpus=None, chunk_length=25):
    """
    Trains LSTM model

    Arguments:
        corpus -- if specified, uses user specification as training data; otherwise, uses books from Project Gutenberg
        chunk_length -- size of strings used for training examples
        model_name -- name of model which will be used to pickle the model later

    Returns:
        model -- returns LSTM model as Sequential object
    
    Effects:
        saves model with name "model_name" parameter with Pickle
    """

    if not corpus:
        corpus = generate_corpus()

    X, y = generate_one_hot(corpus, chunk_length)
    n_chars = sorted(list(set(corpus)))

    model = Sequential()
    model.add(LSTM(128, input_shape=(chunk_length, n_chars)))
    model.add(Dense(n_chars))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.1, batch_size=150, epochs=20, shuffle=True).history
    model.save(model_name)
    pickle.dump(history, open('history.p', 'wb'))
    return model
