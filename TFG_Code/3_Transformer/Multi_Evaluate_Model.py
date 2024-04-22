import matplotlib.dates as mdates
from evaluate import load
from gluonts.time_feature import get_seasonality
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName

# %%


def forecasting(model, test_dataloader):

    accelerator = Accelerator()
    device = accelerator.device

    model.eval()

    forecasts = []

    for batch in test_dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(
                device)
            if model.config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if model.config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs.sequences.cpu().numpy())

    forecasts = np.vstack(forecasts)

    return forecasts


# %%

def see_metrics(forecasts, test_dataset, prediction_length, freq, output_file):
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    forecast_median = np.median(forecasts, 1).squeeze(0).T

    mase_metrics = []
    smape_metrics = []

    with open(output_file, 'w') as f:
        f.write("MASE\t\t\tsMAPE\n")

        for item_id, ts in enumerate(test_dataset):
            training_data = ts["target"][:-prediction_length]
            ground_truth = ts["target"][-prediction_length:]
            mase = mase_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
                training=np.array(training_data),
                periodicity=get_seasonality(freq))
            mase_metrics.append(mase["mase"])

            smape = smape_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
            )
            smape_metrics.append(smape["smape"])

            f.write(f"{mase_metrics[-1]:.6f}\t\t{smape_metrics[-1]:.6f}\n")
    plt.scatter(mase_metrics, smape_metrics, alpha=0.2)
    plt.xlabel("MASE")
    plt.ylabel("sMAPE")
    plt.show()

# %%


def plot(forecasts, ts_index, mv_index, multi_variate_test_dataset, freq, prediction_length):
    # Crea una figura y un eje para el gráfico
    fig, ax = plt.subplots()

    # Genera un índice de tiempo basado en el inicio y la frecuencia del dataset
    index = pd.period_range(
        start=multi_variate_test_dataset[ts_index][FieldName.START],
        periods=10455,
        freq=multi_variate_test_dataset[ts_index][FieldName.START].freq,
    ).to_timestamp()

    # Configura el localizador de horas menores en el eje x
    ax.xaxis.set_minor_locator(mdates.HourLocator())

    # Dibuja la serie temporal real de los últimos '2 * prediction_length' puntos
    ax.plot(
        index[-2 * prediction_length:],
        multi_variate_test_dataset[ts_index]["target"][mv_index, -
                                                       2 * prediction_length:],
        label="actual",
    )

    # Dibuja la media de las predicciones para el último 'prediction_length'
    ax.plot(
        index[-prediction_length:],
        forecasts[ts_index, ..., mv_index].mean(axis=0),
        label="mean",
    )

    # Rellena el área entre la media más/menos una desviación estándar
    ax.fill_between(
        index[-prediction_length:],
        forecasts[ts_index, ..., mv_index].mean(0)
        - forecasts[ts_index, ..., mv_index].std(axis=0),
        forecasts[ts_index, ..., mv_index].mean(0)
        + forecasts[ts_index, ..., mv_index].std(axis=0),
        alpha=0.2,
        interpolate=True,
        label="+/- 1-std",
    )
    ax.legend()
    fig.autofmt_xdate()
