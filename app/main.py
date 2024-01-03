##
import logging
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Embedding, GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from app.config import lstm_settings, settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

plt.style.use("ggplot")


def load_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # load data
    train_data = pd.read_csv(settings.TRAIN_LOCATION)
    test_data = pd.read_csv(settings.TEST_LOCATION)

    # set index
    train_data = train_data.set_index("id", drop=True)
    test_data = test_data.set_index("id", drop=True)

    # get titles of columns with missing values
    logger.info("Fixing data...")
    missing_cols = train_data.columns[train_data.isnull().any()].tolist()
    # remove 'text' from this list:
    missing_cols.remove("text")
    # fill missing values with 'NA', except for the 'text' column
    train_data[missing_cols] = train_data[missing_cols].fillna(value="NA")
    # remove where text is missing:
    train_data = train_data.dropna()
    # drop columns where text length is less than MIN_TEXT_THRESHOLD
    train_data = train_data[train_data["text"].map(len) > settings.MIN_TEXT_THRESHOLD]

    # fill missing test_data:
    test_data = test_data.fillna(value="NA")

    # add text length and title length as features:
    train_data["text_length"] = train_data["text"].map(len)
    train_data["title_length"] = train_data["title"].map(len)
    return train_data, test_data


def generate_test_text(test_data: pd.DataFrame, tokenizer: Tokenizer) -> List[str]:
    logger.info("Generating test text...")
    tokenizer.fit_on_texts(texts=test_data["text"])
    test_text = tokenizer.texts_to_sequences(texts=test_data["text"])
    test_text = pad_sequences(
        sequences=test_text, maxlen=settings.MAX_FEATURES, padding="pre"
    )
    return test_text


def get_tokenizer(training_data: pd.DataFrame) -> Tokenizer:
    # create a tokenizer and fit it on the training data
    logger.info("Generating tokenizer")
    tokenizer = Tokenizer(
        num_words=settings.MAX_FEATURES,
        filters=settings.TOKENIZER_FILTER_LIST,
        lower=True,
        split=" ",
    )
    logger.info("Fitting tokenizer")
    tokenizer.fit_on_texts(training_data["text"])
    return tokenizer


def get_sequences(
    tokenizer: Tokenizer, data: pd.DataFrame
) -> Tuple[List[List[int]], List[List[int]]]:
    logger.info("Generating sequences")
    sequences = tokenizer.texts_to_sequences(data["text"])
    padded_sequences = pad_sequences(
        sequences, maxlen=settings.MAX_FEATURES, padding="pre"
    )
    return sequences, padded_sequences


def generate_split_data(
    X: List[List[int]], Y: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=settings.TEST_SPLIT_SIZE,
        random_state=settings.RANDOM_STATE,
    )
    return X_train, X_test, Y_train, Y_test


# TODO: fine tune
def generate_model(possible_targets: Set[int], tokenizer: Tokenizer) -> Sequential:
    model = Sequential(name=lstm_settings.MODEL_NAME)

    # generate model
    logger.info("Generating model layers")
    embedding_layer = Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=lstm_settings.OUTPUT_DIM,
        name="embedding_layer",
    )
    lstm_layer = LSTM(
        units=lstm_settings.LSTM_UNITS,
        dropout=lstm_settings.LSTM_DROPOUT,
        recurrent_dropout=lstm_settings.LSTM_RECURRENT_DROPOUT,
        name="layer_2",
    )
    dropout_layer_initial = Dropout(rate=lstm_settings.DROPOUT_RATE, name="layer_3")
    dense_layer_large = Dense(
        units=lstm_settings.DENSE_UNITS_LARGE,
        activation=lstm_settings.DENSE_ACTIVATION,
        name="large_dense_layer",
    )
    dense_layer_small = Dense(
        units=lstm_settings.DENSE_UNITS_SMALL,
        activation=lstm_settings.DENSE_ACTIVATION,
        name="small_dense_layer",
    )
    dropout_layer_final = Dropout(rate=lstm_settings.DROPOUT_RATE, name="layer_5")
    output_layer = Dense(
        units=len(possible_targets),
        activation=lstm_settings.OUTPUT_ACTIVATION,
        name="output_layer",
    )
    logger.info("Adding layers to model")
    for layer in [
        embedding_layer,
        GlobalAveragePooling1D(),
        dense_layer_large,
        dense_layer_small,
        output_layer,
    ]:
        model.add(layer=layer)

    return model


def plot_accuracy(history) -> None:
    plt.figure()

    plt.plot(
        np.arange(0, lstm_settings.EPOCHS),
        1 - np.array(history.history["loss"]),
        label="Training Accuracy",
    )
    plt.plot(
        np.arange(0, lstm_settings.EPOCHS),
        1 - np.array(history.history["val_loss"]),
        label="Validation Accuracy",
    )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # first we get the data
    train_data, test_data = load_clean_data()
    logger.info(
        f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}"
    )

    settings.MAX_FEATURES = int(train_data["text_length"].mean())
    lstm_settings.INPUT_DIM = settings.MAX_FEATURES

    # then get the tokenizers and sequences
    tk = get_tokenizer(train_data)
    sequences, padded_sequences = get_sequences(tk, train_data)
    lstm_settings.INPUT_LENGTH = max(len(sequence) for sequence in sequences)

    # split the data
    X_train, X_test, Y_train, Y_test = generate_split_data(
        X=padded_sequences, Y=train_data["label"].values
    )

    # generate and fit the model
    lstm_model = generate_model(
        possible_targets=set(train_data["label"].values), tokenizer=tk
    )
    logger.info("Model received, compiling")
    lstm_model.compile(
        loss=lstm_settings.LOSS,
        optimizer=lstm_settings.OPTIMIZER,
        metrics=lstm_settings.COMPILATION_METRICS,
    )
    logger.info("Model compiled, fitting")
    lstm_model_fit = lstm_model.fit(
        x=X_train,
        y=Y_train,
        epochs=lstm_settings.EPOCHS,
        batch_size=lstm_settings.BATCH_SIZE,
        validation_split=lstm_settings.VALIDATION_SPLIT,
    )

    # make predictions
    test_text = generate_test_text(test_data, tk)
    logger.info("Predicting")
    predictions = lstm_model.predict(test_text)
    prediction_classes = np.argmax(predictions, axis=1)
    logger.info("Predictions made")

    # plot accuracy
    # I get ~0.9 accuracy after 5 epochs
    plot_accuracy(lstm_model_fit)
