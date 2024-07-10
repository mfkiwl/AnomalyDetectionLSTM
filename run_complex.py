"""_summary_
@author: outisa
"""
# %%´¨
import time
import random
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from utils.helper_functions import epoch_time, print_model_params, to_test_sequences, to_train_sequences
from utils.prediction_visual import anomaly_calc
from utils.handle_data import read_data_from_dat
from utils.train import train_model
from utils.predict import predict
from CVAutoencoder import ComplexAutoencoder, custom_mse_criterion

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

# %%
# reate sequences for training data
data = read_data_from_dat('your_path_dat/*.DAT', WINDOW)

#%%
train_data, list_of_endings_tr, train_class = [], [] ,[]
to_train_sequences(data, train_data, train_class, list_of_endings_tr, LEN_SEQ)

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
to_test_sequences(test_data, testData, LEN_SEQ, list_of_endings)
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
            threshold_max = parts[2].split(':')[1]
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
