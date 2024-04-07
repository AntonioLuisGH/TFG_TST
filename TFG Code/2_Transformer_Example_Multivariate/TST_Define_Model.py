# %% LIBRARIES

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import get_lags_for_frequency
from pandas.core.arrays.period import period_array
import pandas as pd

# %% DEFINE MODEL


def define_my_model(num_of_variates, multi_variate_train_dataset, freq, prediction_length):

    # Let's use the default lags provided by GluonTS for the given frequency:
    lags_sequence = get_lags_for_frequency(freq)
    # Let's also check the default time features that GluonTS provides us:
    time_features = time_features_from_frequency_str(freq)
    # we'll add these features as a scalar values
    timestamp = pd.Period("2015-01-01 01:00:01", freq=freq)
    timestamp_as_index = pd.PeriodIndex(data=period_array([timestamp]))
    additional_features = [
        (time_feature.__name__, time_feature(timestamp_as_index))
        for time_feature in time_features
    ]

    # Define config
    config = TimeSeriesTransformerConfig(
        # in the multivariate setting, input_size is the number of variates in the time series per time step:
        input_size=num_of_variates,
        # prediction length:
        prediction_length=prediction_length,
        # context length:
        context_length=prediction_length * 2,
        # lags value copied from 1 week before:
        lags_sequence=[1, 24 * 7],
        # we'll add 2 time features ("month of year" and "age", see further):
        num_time_features=len(time_features) + 1,

        # transformer params:
        encoder_layers=4,
        decoder_layers=4,
        d_model=32,
    )

    model = TimeSeriesTransformerForPrediction(config)

    return model
