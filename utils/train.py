import time
import torch
import numpy as np
import copy
from utils.helper_functions import epoch_time
from utils.prediction_visual import plot_train_losses

def train_model(model, train_x, val_x, criterion, optimizer, batch_size, n_epochs, device, model_path):
    train_loss, val_loss = [], []
    best_vloss = 1000
    count = 0
    with open(f'{model_path}_threshold_values.txt', 'w') as best_vals:
        best_vals.write(f'{model_path}\n')
        for epoch in range(n_epochs):
            if epoch == 30:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            start_time = time.time()
            train_losses, val_losses = [], []
            model.train(True)
            for i in range(0, len(train_x), batch_size):
                optimizer.zero_grad()
                # Batch first: Batch size, sequence length, features
                train_seq = torch.reshape(
                    torch.stack(train_x[i:i+batch_size]), (len(train_x[i:i+batch_size]), train_x[i].shape[0], train_x[i].shape[1])).to(device)
                seq_pred = model(train_seq)
                loss = criterion(seq_pred, train_seq)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            mean_loss_train = np.mean(train_losses)
            train_loss.append(mean_loss_train)
        
            with torch.no_grad():
                model.eval()
                for i in range(0, len(val_x), batch_size):
                    val_seq = torch.reshape(
                        torch.stack(val_x[i:i+batch_size]), (len(val_x[i:i+batch_size]), val_x[i].shape[0], val_x[i].shape[1])).to(device)
                    seq_pred = model(val_seq)
                    loss = criterion(seq_pred, val_seq)
                    val_losses.append(loss.item())
                mean_loss = np.mean(val_losses)
                val_loss.append(mean_loss)
                if mean_loss < best_vloss:
                    best_vloss = mean_loss
                    to_add = f'Loss_values mean:{mean_loss:.6f} max:{np.max(val_losses):.6f} min:{np.min(val_losses):6f}\n'
                    best_vals.write(to_add)
                    torch.save(model.state_dict(), model_path)
                    best_model = copy.deepcopy(model)
                elif mean_loss > best_vloss:
                    count += 1
                    if count >= 3:
                        break

            end_time = time.time()
            epoch_time(start_time, end_time)
            print(f'Epoch {epoch}: train loss: {mean_loss_train:.6f} val loss: {mean_loss:.6f}')
        to_add = to_add.replace('\n', '')
        best_vals.write(to_add)
    plot_train_losses(val_loss, train_loss)
    return best_model
