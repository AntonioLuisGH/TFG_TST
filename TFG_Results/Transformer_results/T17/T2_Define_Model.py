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

    # Default lags provided by GluonTS for the given frequency:
    lags_sequence = get_lags_for_frequency(freq)
    # Default time features that GluonTS provides us:
    time_features = time_features_from_frequency_str(freq)

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
            lags_sequence=[1, 2, 3, 4, 5, 183+1, 183+2, 183+3, 183+4, 183+5, 366+1, 366+2, 366+3, 366+4, 366+5, 549+1, 549+2, 549+3, 549+4, 549+5],
            # we'll add 2 time features ("month of year" and "age", see further):
            num_time_features=len(time_features) + 1,

            # transformer params:
            dropout=0.2,
            encoder_layers=2,
            decoder_layers=2,
            d_model=16,
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
            lags_sequence=[1],
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
            lags_sequence=[1, 2, 3, 4, 5, 184, 184+1, 184+2, 184+3, 184+4, 184+5, 368, 368+1, 368+2, 368+3, 368+4, 368+5],
            # we'll add 5 time features ("hour_of_day", ..., and "age"):
            num_time_features=len(time_features) + 1,

            # autoformer params:
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
