"""_summary_
@author: outisa
"""
# %%´¨
import time
import random
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from utils.weight_init import initialize_weights
from utils.helper_functions import (epoch_time, print_model_params)
from utils.prediction_visual import anomaly_calc
from utils.handle_data import read_data_from_dat
from utils.train import train_model
from utils.predict import predict

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 150
LR = 0.0001
BATCHSIZE = 128
NUM_LAYERS = 1
FEATURES = 1024
HIDDEN_DIM = 450
LEN_SEQ = 4
WINDOW = FEATURES
model_path = f'best_cv_model_{datetime.datetime}.pt'

# Autoencoder
class ComplexAutoencoder(nn.Module):
    def __init__(self, features_n, hidden_dim, num_layers, seq_length) -> None:
        super().__init__()
        # expected features, hidden sizes, number of stacked layers, batch_first=True means that
        # input shape should be N,L,H_in -> batch size, sequence length, input size
        self.features_n = features_n
        self.seq_len = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm_in1 = nn.LSTM(self.features_n, self.hidden_dim*2,
            num_layers, batch_first=True, bias=True).to(torch.cdouble)
        self.lstm_in2 = nn.LSTM(self.hidden_dim*2, self.hidden_dim,
            num_layers, batch_first=True, bias=True).to(torch.cdouble)
        self.lstm_out1 = nn.LSTM(self.hidden_dim, self.hidden_dim*2,
            num_layers, batch_first=True, bias=True).to(torch.cdouble)
        self.lstm_out2 = nn.LSTM(self.hidden_dim*2, self.features_n,
            num_layers, batch_first=True, bias=True).to(torch.cdouble)
        self.linear = nn.Linear(
            self.features_n, self.features_n, bias=True).to(torch.cdouble)

        self.init_hidden(self.lstm_in1, self.features_n, self.hidden_dim*2)
        self.init_hidden(self.lstm_in2, self.hidden_dim*2, self.hidden_dim)
        self.init_hidden(self.lstm_out1, self.hidden_dim, self.hidden_dim*2)
        self.init_hidden(self.lstm_out2, self.hidden_dim*2, self.features_n)
        self.init_linear_weights(self.linear, self.features_n, self.features_n)
        self.init_linear_weights(self.linear1, self.hidden_dim*2, self.hidden_dim*2)
    # Initialize hidden states
    def init_hidden(self, mod, dim1, dim2):
        for value in mod.state_dict():
            # Input-hidden weights at first cell layer.
            if 'weight_ih_l0' in value:
                weight_ih_data_ii = initialize_weights(dim1, dim2)  # I_Wii
                weight_ih_data_if = initialize_weights(dim1, dim2)  # I_Wif
                weight_ih_data_ic = initialize_weights(dim1, dim2)  # I_Wic
                weight_ih_data_io = initialize_weights(dim1, dim2)  # I_Wio
                weight_ih_data = torch.stack(
                    [weight_ih_data_ii, weight_ih_data_if, weight_ih_data_ic, weight_ih_data_io],
                    dim=0
                )
                # 4*hidden, input
                weight_ih_data = weight_ih_data.view(dim2*4, dim1)
                mod.state_dict()[value].data.copy_(
                    weight_ih_data).requires_grad_()
            # Input-hidden weights at the second and upcoming layers
            elif 'weight_ih' in value:
                weight_ih_data_ii = initialize_weights(dim2, dim2)  # I_Wii
                weight_ih_data_if = initialize_weights(dim2, dim2)  # I_Wif
                weight_ih_data_ic = initialize_weights(dim2, dim2)  # I_Wic
                weight_ih_data_io = initialize_weights(dim2, dim2)  # I_Wio
                weight_ih_data = torch.stack(
                    [weight_ih_data_ii, weight_ih_data_if, weight_ih_data_ic, weight_ih_data_io],
                    dim=0
                )
                # 4*hidden, hidden
                weight_ih_data = weight_ih_data.view(dim2*4, dim2)
                mod.state_dict()[value].data.copy_(
                    weight_ih_data).requires_grad_()
            # Hidden to hidden layers
            elif 'weight_hh' in value:
                weight_hh_data_hi = initialize_weights(dim2, dim2)  # H_Whi
                weight_hh_data_hf = initialize_weights(dim2, dim2)  # H_Whf
                weight_hh_data_hc = initialize_weights(dim2, dim2)  # H_Whc
                weight_hh_data_ho = initialize_weights(dim2, dim2)  # H_Who
                weight_hh_data = torch.stack(
                    [weight_hh_data_hi, weight_hh_data_hf, weight_hh_data_hc, weight_hh_data_ho],
                    dim=0
                )
                # 4*hidden, hidden
                weight_hh_data = weight_hh_data.view(dim2*4, dim2)
                mod.state_dict()[value].data.copy_(
                    weight_hh_data).requires_grad_()
            # Bias for both hidden-hidden and input-hidden layers
            elif 'bias' in value:
                bias_i = initialize_weights(dim2, 1) * 0
                bias_f = initialize_weights(dim2, 1)
                bias_c = initialize_weights(dim2, 1) * 0
                bias_o = initialize_weights(dim2, 1) * 0
                bias = torch.stack([bias_i, bias_f, bias_c, bias_o])
                bias = bias.view(dim2*4)
                mod.state_dict()[value].data.copy_(bias).requires_grad_()

    # Initialize weights for linear unit
    def init_linear_weights(self, mod, dim1, dim2):
        for value in mod.state_dict():
            if 'weight' in value:
                weight = initialize_weights(dim1, dim2)
                mod.state_dict()[value].data.copy_(weight).requires_grad_()

    def forward(self, inputs):
        output, _ = self.lstm_in1(inputs)
        output, _ = self.lstm_in2(output)
        output, _ = self.lstm_out1(output)
        output, _ = self.lstm_out2(output)
        return self.linear(output)

def custom_mse_criterion(pred, target):
    return torch.mean(torch.abs(pred-target)**2)

# %%
# reate sequences for training data
data = read_data_from_dat('your_path_dat/*.DAT', WINDOW)
print(data.keys())

#%%
train_data, list_of_endings1, train_class = [], [] ,[]
for key in data.keys():
    print(len(data[key]))
    for i in range(0, len(data[key]), LEN_SEQ):
        if len(data[key][i:i+LEN_SEQ]) == LEN_SEQ:
            train_data.append(data[key][i:i+LEN_SEQ])
            train_class.append(key)
    list_of_endings1.append(key)

print(len(train_data))

MODEL = ComplexAutoencoder(FEATURES, HIDDEN_DIM,
    NUM_LAYERS, LEN_SEQ).to(DEVICE)
optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)

start_total = time.time()
train_set, val_set, train_y, val_y = train_test_split(train_data, train_class, test_size=0.2, random_state=42)
criterion = custom_mse_criterion
(
    trained_model,
    train_losses_final,
    val_losses_final,
) = train_model(
    MODEL,
    train_set,
    val_set,
    criterion,
    optimizer,
    BATCHSIZE,
    EPOCHS,
    DEVICE,
    WINDOW,
    model_path
)
end_total = time.time()
epoch_time(start_total, end_total)
print_model_params(trained_model)

# %%
MODEL = ComplexAutoencoder(FEATURES,HIDDEN_DIM, NUM_LAYERS, LEN_SEQ).to(DEVICE)
#optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
criterion = custom_mse_criterion
MODEL.load_state_dict(torch.load('model_path'))
MODEL.eval()
testData = read_data_from_dat('your_path/*.DAT', WINDOW)

#%%
test_data, list_of_endings= [], []
end_sample = 0
for key in testData.keys():
    for i in range(0, len(testData[key]), LEN_SEQ):
        if len(testData[key][i:i+LEN_SEQ]) == LEN_SEQ:
            test_data.append(testData[key][i:i+LEN_SEQ])
    end_sample = len(test_data)        
    list_of_endings.append((key, end_sample))
#%%
start = time.time()
loss_vals, predicted_vals = predict(MODEL, val_set, LEN_SEQ, FEATURES, DEVICE, criterion)
end = time.time()
epoch_time(start, end)
#%%
threshold_max, threshold_min = 0, 0

with open('threshold_values.txt', 'r') as best_vals:
    correct_point = False
    for line in best_vals:
        if correct_point:
            parts = line.split(' ')
            threshold_max = parts[3].split(':')[1]
            threshold_min = parts[3].split(':')[2]
            break
        if line.startswith(model_path):
            correct_point =  True
    best_vals.close()
font = {'size': 28}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (27, 20)
df_to_plot = pd.DataFrame({'losses': loss_vals})
df_to_plot['index'] = df_to_plot.index
df_to_plot['category'] = 'Clean'

plt.clf()
plt.title('Loss values of the different types of GNSS signal (CVAutoencoder)')
sns.scatterplot(data=df_to_plot, hue='category', style='category', x='index', y='losses', s=50)
plt.axhline(y = threshold_max, color = 'red', label = 'Threshold_max')
plt.axhline(y = threshold_min, color = 'blue', label = 'Threshold_min')
plt.xlabel('Test sample')
plt.ylabel('Mean Absolute Error')
plt.ylim(0,0.09)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

print(len(loss_vals))
anomaly_calc(loss_vals, threshold_max, threshold_min, len(loss_vals), list_of_endings)

# %%
