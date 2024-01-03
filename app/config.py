from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Implements the settings"""

    TRAIN_LOCATION: str = "data/train.csv"
    TEST_LOCATION: str = "data/test.csv"

    MIN_TEXT_THRESHOLD: int = 30

    TOKENIZER_FILTER_LIST: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    MAX_FEATURES: int = 0

    TEST_SPLIT_SIZE: float = 0.2
    RANDOM_STATE: int = 20190427

    LOG_LEVEL: str = "INFO"


class LSTMSettings(BaseSettings):
    """Implements the settings"""

    # model generation
    MODEL_NAME: str = "lstm"

    # Embedding
    INPUT_DIM: int = 0
    INPUT_LENGTH: int = 0
    OUTPUT_DIM: int = 100

    LSTM_UNITS: int = 100
    LSTM_DROPOUT: float = 0.2
    LSTM_RECURRENT_DROPOUT: float = 0.2
    DROPOUT_RATE: float = 0.2

    # Dense
    DENSE_UNITS_LARGE: int = 120
    DENSE_UNITS_SMALL: int = 24
    DENSE_ACTIVATION: str = "relu"

    # Output
    OUTPUT_ACTIVATION: str = "softmax"

    # Compilation
    LOSS: str = "sparse_categorical_crossentropy"
    OPTIMIZER: str = "adam"
    COMPILATION_METRICS: List[str] = ["accuracy"]

    # fitting
    EPOCHS: int = 5
    BATCH_SIZE: int = 64
    VALIDATION_SPLIT: float = 0.2


settings = Settings()
lstm_settings = LSTMSettings()
