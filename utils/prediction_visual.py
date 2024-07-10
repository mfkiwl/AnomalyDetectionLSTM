"""_summary_
    Visualising results
"""
import numpy as np
import matplotlib.pyplot as plt

def anomaly_calc(losses, threshold_max, threshold_min, samples, list_of_endings):
    anomalies = np.zeros((samples,))
    clean = np.zeros((samples,))
    i = 0
    for loss in losses:
        if loss > threshold_max or loss < threshold_min:
            anomalies[i] = 1
        else:
            clean[i] = 1
        i += 1
    start = 0
    for obj in  list_of_endings:
        print(f"{obj[0]}:\n clean: {np.count_nonzero(clean[start:obj[1]])}\n anomalies: {np.count_nonzero(anomalies[start:obj[1]])}") 
        start = obj[1]

    print(f'Total number of clean samples: \n {np.count_nonzero(clean)}')
    print(f'Total number of anomalies: \n {np.count_nonzero(anomalies)}')

def plot_train_losses(validation_loss_hist, train_loss_hist):
    font = {'size': 22}
    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = (20,15)
    plt.clf()
    plt.plot(validation_loss_hist, color='blue', label='Validation set')
    plt.plot(train_loss_hist, color='red', label='Train set')
    plt.xlabel('Epochs', fontsize=36)
    plt.ylabel('Mean Squared Error', fontsize=36)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.legend(loc='upper right', labels=['Validation set', 'Train set'], fontsize=36)
    plt.ylim(0, 0.16)
    plt.show()

def sanity_check(predicted):
    font = {'size': 26}
    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = (20,15)
    plt.clf()
    plt.suptitle('Sanity check sample 0', fontsize=24)
    plt.subplot(2,1,1)
    plt.title('Real Units')
    plt.plot(predicted[0][0][:].real, color='lightblue')
    plt.subplot(2,1,2)
    plt.title('Imaginary Units')
    plt.plot(predicted[0][0][:].imag, color='coral')

    plt.show()
    plt.clf()
    plt.suptitle('Sanity check sample 100', fontsize=24)
    plt.subplot(2,1,1)
    plt.title('Real Units')
    plt.plot(predicted[100][0][:].real, color='lightblue')
    plt.subplot(2,1,2)
    plt.title('Imaginary Units')
    plt.plot(predicted[100][0][:].imag, color='coral')
    plt.show()

def plot_clean_samples(test_data, predicted, end, list_of_endings):
    start = 0
    for obj in list_of_endings:
        if 'soi' in obj[0].lower():
            break
        start = obj[1]
    if len(test_data) < end:
        end = len(test_data)

    for i in range(start, end):
        plt.clf()
        plt.title('Clean Data (abs-values)')
        plt.subplot(2,2,1)
        plt.title('Real units input')
        plt.plot(test_data[i][0].real)
        plt.subplot(2,2,2)
        plt.title('Imaginary units input')
        plt.plot(test_data[i][0].imag)
        plt.subplot(2,2,3)
        plt.title('Real units predicted')
        plt.plot(predicted[i][0].real)
        plt.subplot(2,2,4)
        plt.title('Imaginary units predicted')
        plt.plot(predicted[i][0].imag)
        plt.show()

def plot_sample(test_data, predicted, idx, title):
    font = {'size': 22}
    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = (20,15)
    plt.clf()
    plt.suptitle(title, fontsize=24)
    plt.subplot(2,2,1)
    plt.title('Real units input')
    plt.plot(test_data[idx][0].real,color='lightblue')
    plt.subplot(2,2,2)
    plt.title('Imaginary units input')
    plt.plot(test_data[idx][0].imag,color='coral')
    plt.subplot(2,2,3)
    plt.title('Real units predicted')
    plt.plot(predicted[idx][0].real,color='lightblue')
    plt.subplot(2,2,4)
    plt.title('Imaginary units predicted')
    plt.plot(predicted[idx][0].imag, color='coral')
    plt.show()

def first_sample_from_class(test_data, predicted, list_of_endings):
    start = 0
    for obj in list_of_endings:
        plot_sample(test_data, predicted, start, obj[0])
        start = obj[1]
