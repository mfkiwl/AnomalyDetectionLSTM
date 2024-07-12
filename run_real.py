#%% 
import time
import torch
from  torch import nn
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helper_functions import epoch_time, to_train_sequences, to_test_sequences
from utils.handle_data import read_data_from_dat
from utils.prediction_visual import anomaly_calc, plot_prediction_losses
from utils.train import train_model
from utils.predict import predict
from real_lstm_autoencoder import RealAutoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Variables
EPOCHS = 20
LR = 0.0001
BATCHSIZE = 16
NUM_LAYERS = 1
WINDOW = 128
FEATURES = WINDOW
HIDDEN_DIM = 100
LEN_SEQ = 4
STEP_SIZE = 1
BANDWIDTH = int(25e6)
sample_interval = 200
real_valued = True
model_path = f'best_rv_model_{datetime.datetime}.pt'

data = {}
read_data_from_dat(data, 'your_path_dat/*.DAT', WINDOW, STEP_SIZE, BANDWIDTH, sample_interval, LEN_SEQ, real_valued)#%%
train_data, list_of_endings_tr, train_class = [], [] ,[]
to_train_sequences(data, train_data, train_class, list_of_endings_tr, LEN_SEQ)
#%%
model = RealAutoencoder(FEATURES,HIDDEN_DIM, NUM_LAYERS, LEN_SEQ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss().to(device)
start_total = time.time()
train_set, val_set, train_y, val_y = train_test_split(train_data, train_class, test_size=0.2, random_state=42)

(
    trained_model,
    train_losses_final,
    val_losses_final,
) = train_model(
    model,
    train_set,
    val_set,
    criterion,
    optimizer,
    BATCHSIZE,
    EPOCHS,
    device,
    model_path
)
end_total = time.time()
epoch_time(start_total, end_total)

# %%
testData = read_data_from_dat('your_data', WINDOW, False)
test_data, list_of_endings = [], []
to_test_sequences(test_data, testData, LEN_SEQ, list_of_endings)
#%%
start = time.time()
loss_vals, predicted_vals = predict(
    model, test_data, LEN_SEQ, FEATURES, device, criterion, dtype=np.float64)
end = time.time()
epoch_time(start, end)

#%%
threshold_max, threshold_min = 0, 0
with open('threshold_values.txt', 'r') as best_vals:
    correct_point = False
    for line in best_vals:
        if correct_point:
            parts = line.split(' ')
            threshold_max = parts[2].split(':')[1]
            break
        if line.startswith(model_path):
            correct_point =  True
    best_vals.close()

plot_prediction_losses(list_of_endings, threshold_max, threshold_min, loss_vals)
anomaly_calc(loss_vals, threshold_max, threshold_min, len(loss_vals), list_of_endings)
