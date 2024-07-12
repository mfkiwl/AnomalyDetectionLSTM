import torch
import numpy as np

def predict(trained_model, test_x, seq_len, features, device, criterion, dtype):
    losses = []
    predicted = np.ndarray((len(test_x), seq_len, features), dtype=dtype)
    with torch.no_grad():
        trained_model.eval()
        len_test = len(test_x)
        for i in range(len_test):
            seq_true = torch.reshape(
                test_x[i], (1, test_x[i].shape[0], test_x[i].shape[1])).to(device)
            seq_pred = trained_model(seq_true)
            loss = criterion(seq_pred, seq_true)
            losses.append(loss.item())
            predicted[i] = seq_pred
        return losses, predicted
