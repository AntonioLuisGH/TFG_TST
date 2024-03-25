# %% LIBRARIES

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import get_lags_for_frequency

# %% DEFINE MODEL


def define_my_model(train_dataset, freq, prediction_length):

    # Let's use the default lags provided by GluonTS for the given frequency:
    lags_sequence = get_lags_for_frequency(freq)
    # Let's also check the default time features that GluonTS provides us:
    time_features = time_features_from_frequency_str(freq)

    # Define config
    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        # context length:
        context_length=prediction_length * 2,
        # lags coming from helper given the freq:
        lags_sequence=lags_sequence,
        # we'll add 2 time features ("month of year" and "age", see further):
        num_time_features=len(time_features) + 1,
        # we have a single static categorical feature, namely time series ID:
        num_static_categorical_features=1,
        # it has 366 possible values:
        cardinality=[len(train_dataset)],
        # the model will learn an embedding of size 2 for each of the 366 possible values:
        embedding_dimension=[2],

        # transformer params:
        encoder_layers=4,
        decoder_layers=4,
        d_model=32,
    )

    model = TimeSeriesTransformerForPrediction(config)

    return model
