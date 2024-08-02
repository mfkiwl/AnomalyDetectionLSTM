"""_summary_
@author: outisa
"""
# %%´¨
import time
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
from utils.helper_functions import epoch_time, write_model_params, to_test_sequences, to_train_sequences
from utils.prediction_visual import anomaly_calc, plot_prediction_losses
from utils.handle_data import read_data_from_dat
from utils.train import train_model
from utils.predict import predict
from CVAutoencoder import ComplexAutoencoder, custom_mse_criterion

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 150
LR = 0.0001
BATCHSIZE = 128
NUM_LAYERS = 1
FEATURES = 1024
HIDDEN_DIM = 450
LEN_SEQ = 4
WINDOW = FEATURES
BANDWIDTH = int(25e6)
sample_length = 200
sample_interval = int(2e-4)
model_path = f'best_cv_model_{datetime.datetime.now()}.pt'

# %%
# reate sequences for training data
data = {}
read_data_from_dat(data, 'your_path_dat/*.DAT', WINDOW, BANDWIDTH, sample_length, sample_interval)

#%%
train_data, list_of_endings_tr, train_class = [], [] ,[]
to_train_sequences(data, train_data, train_class, list_of_endings_tr, LEN_SEQ)

MODEL = ComplexAutoencoder(FEATURES, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)

start_total = time.time()
train_set, val_set, train_y, val_y = train_test_split(train_data, train_class, test_size=0.2, random_state=42)
criterion = custom_mse_criterion
trained_model = train_model(
    MODEL,
    train_set,
    val_set,
    criterion,
    optimizer,
    BATCHSIZE,
    EPOCHS,
    DEVICE,
    model_path
)
end_total = time.time()
epoch_time(start_total, end_total)
write_model_params(trained_model, model_path)
# %%
# in case of need, get the right model. note: use correct hyperparameters
# trained_model = ComplexAutoencoder(FEATURES, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
# trained_model.load_state_dict(torch.load(model_path)) # change correct model path
# trained_model.eval()
testData = {}
read_data_from_dat(testData, 'your_path_dat/*.DAT', WINDOW, BANDWIDTH, sample_length, sample_interval)

test_data, list_of_endings= [], []
to_test_sequences(test_data, testData, LEN_SEQ, list_of_endings)

start = time.time()
loss_vals, predicted_vals = predict(MODEL, val_set, LEN_SEQ, FEATURES, DEVICE, criterion, dtype=np.complex128)
end = time.time()
epoch_time(start, end)

threshold_max, threshold_min = 0, 0
with open(f'{model_path}_threshold_values.txt', 'r') as best_vals:
    lines = best_vals.readlines()
    parts = lines[-1].split(' ')
    print(parts)
    threshold_max = float(parts[2].split(':')[1])
    threshold_min = float(parts[3].split(':')[1])

plot_prediction_losses(list_of_endings, threshold_max, threshold_min, loss_vals)
anomaly_calc(loss_vals, threshold_max, threshold_min, len(loss_vals), list_of_endings)
