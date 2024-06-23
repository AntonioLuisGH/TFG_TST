# %% LIBRARIES

from transformers import InformerConfig, InformerForPrediction
from transformers import AutoformerConfig, AutoformerForPrediction
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import get_lags_for_frequency
from pandas.core.arrays.period import period_array
import pandas as pd

# %% DEFINE MODEL


def define_my_model(num_of_variates, multi_variate_train_dataset, model_variant, freq, prediction_length):

    # Let's use the default lags provided by GluonTS for the given frequency:
    lags_sequence = get_lags_for_frequency(freq)
    # Let's also check the default time features that GluonTS provides us:
    time_features = time_features_from_frequency_str(freq)
    # we'll add these features as a scalar values
    timestamp = pd.Period("2020-10-01 01:06:48", freq=freq)
    timestamp_as_index = pd.PeriodIndex(data=period_array([timestamp]))
    additional_features = [
        (time_feature.__name__, time_feature(timestamp_as_index))
        for time_feature in time_features
    ]

    if (model_variant == "Transformer"):
        # Define config
        config = TimeSeriesTransformerConfig(
            # in the multivariate setting, input_size is the number of variates in the time series per time step:
            input_size=num_of_variates,
            # prediction length:
            prediction_length=prediction_length,
            # context length:
            context_length=prediction_length * 2,
            # lags value copied from ... week before:
            lags_sequence=[1, 24*7],
            # we'll add 2 time features ("month of year" and "age", see further):
            num_time_features=len(time_features) + 1,

            # transformer params:
            dropout=0.1,
            encoder_layers=4,
            decoder_layers=4,
            d_model=32,
        )
        model = TimeSeriesTransformerForPrediction(config)

    elif (model_variant == "Informer"):
        # Define config
        config = InformerConfig(
            # in the multivariate setting, input_size is the number of variates in the time series per time step
            input_size=num_of_variates,
            # prediction length:
            prediction_length=prediction_length,
            # context length:
            context_length=prediction_length * 2,
            # lags value copied from 1 week before:
            lags_sequence=[1, 24*7],
            # we'll add 5 time features ("hour_of_day", ..., and "age"):
            num_time_features=len(time_features) + 1,

            # informer params:
            dropout=0.1,
            encoder_layers=6,
            decoder_layers=4,
            # project input from num_of_variates*len(lags_sequence)+num_time_features to:
            d_model=64,
        )
        model = InformerForPrediction(config)

    elif (model_variant == "Autoformer"):
        # Define config
        config = AutoformerConfig(
            # in the multivariate setting, input_size is the number of variates in the time series per time step
            input_size=num_of_variates,
            # prediction length:
            prediction_length=prediction_length,
            # context length:
            context_length=prediction_length * 2,
            # lags value copied from 1 week before:
            lags_sequence=[1, 2, 3, 4, 180, 181, 182, 183, 184, 360, 361, 362, 363, 364],
            # we'll add 5 time features ("hour_of_day", ..., and "age"):
            num_time_features=len(time_features) + 1,

            # informer params:
            dropout=0.1,
            encoder_layers=6,
            decoder_layers=4,
            # project input from num_of_variates*len(lags_sequence)+num_time_features to:
            d_model=64,
        )
        model = AutoformerForPrediction(config)

    else:
        print("ERROR: No valid variant")

    return model
