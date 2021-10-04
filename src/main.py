import numpy as np
#from train_model import generate_corpus
#from train_model import train_LTSM_model
from keras.models import load_model

def main():
    model = load_model('../models/lstm_model')
    while True:
        text = input()
        if text == "end":
            break

if __name__ == "__main__":
    main()
    