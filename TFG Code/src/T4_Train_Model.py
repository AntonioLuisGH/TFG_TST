from accelerate import Accelerator
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np


def train_model(num_of_epochs, model, train_dataloader):
    epochs = num_of_epochs
    loss_history = []

    accelerator = Accelerator()
    device = accelerator.device

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=6e-4,
                      betas=(0.9, 0.95), weight_decay=1e-1)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    model.train()
    for epoch in range(epochs):
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(
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
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            loss_history.append(loss.item())
            if idx % 100 == 0:
                print(loss.item())

    # view training
    loss_history = np.array(loss_history).reshape(-1)
    x = range(loss_history.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(x, loss_history, label="train")
    plt.title("Loss", fontsize=15)
    plt.legend(loc="upper right")
    plt.xlabel("iteration")
    plt.ylabel("nll")
    plt.show()

    return model
