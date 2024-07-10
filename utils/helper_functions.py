import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch, hann
import pandas as pd
import seaborn as sns

def scaling(fft_data):
    """_summary_

    Args:
        X (ndarray): an ndarray of complex numbers

    Returns:
        ndarray: Scaled ndarray of complex values
    """
    scaler = torch.max(abs(fft_data.real))
    if scaler < torch.max(torch.abs(fft_data.imag)):
        scaler = torch.max(torch.abs(fft_data.imag))

    if (-1 < torch.max(torch.abs(fft_data.real/scaler)) < 1) and (-1 < torch.max(torch.abs(fft_data.imag/scaler)) < 1):
        print(fft_data.real/scaler, fft_data.imag/scaler)

    return  fft_data.real/scaler + 1j*fft_data.imag/scaler

def scaling_same_scaler(fft_data, window_len):
    return  fft_data.real/window_len + 1j*fft_data.imag/window_len

def epoch_time(start_time, end_time):
    """_summary_

    Args:
        start_time: Starting time in milliseconds.
        end_time: Ending time in milliseconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    elapsed_ms = (elapsed_time - int(elapsed_time)) *1000
    print(f'total time: {elapsed_mins} mins {elapsed_secs} s  {elapsed_ms} ms')

def to_train_sequences(df, train_data, train_class, list_of_endings_tr, len_seq):
    for key in df.keys():
        print(len(df[key]))
        for i in range(0, len(df[key]), len_seq):
            if len(df[key][i:i+len_seq]) == len_seq:
                train_data.append(df[key][i:i+len_seq])
                train_class.append(key)
        list_of_endings_tr.append(key)
    print(len(train_data))

def to_test_sequences(test_data, df, len_seq, list_of_endings):
    print(df.keys())
    for key in df.keys():
        for i in range(0, len(df[key]), len_seq):
            if len(test_data[key][i:i+len_seq]) == len_seq:
                test_data.append(test_data[key][i:i+len_seq])
        end_sample = len(test_data)        
        list_of_endings.append((key, end_sample))
    print(len(test_data))

def plot_magnitude(data, title):
    plt.clf()
    plt.figure(figsize=(20,15))
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.title(title, fontsize=36)
    plt.plot(10*torch.log10(abs(data)))
    plt.xlabel('Frequency features', fontsize=36)
    plt.ylabel('Magnitude in dB', fontsize=36)
    plt.ylim(0,150)
    plt.show()

def plot_IQ(col, title):
    plt.clf()
    plt.figure(figsize=(20,15))
    plt.tick_params(axis='both', which='major', labelsize=36)
    plt.title(title, fontsize=36)
    plt.plot(np.real(col),color='blue')
    plt.plot(np.imag(col),color='coral')
    plt.xlabel('Frequency features', fontsize=36)
    plt.ylabel('Amplitude', fontsize=36)
    plt.ylim(-10,10)
    plt.legend(['Real', 'Imaginary'], fontsize=36)
    plt.show()

def plot_complex_vectors(input_vals, title, window_len, polar_wanted=False):
    """ Plots given sequence and subsequence of it in cartesian coordinates.
shot. 
    Args:
        input_vals (list): List of complex values to be plotted
        start (Integer): Starting index for the plotted subsequence.
        end (Integer): Ending index for the plotted subsequence.
    """
    font = {'size': 24}
    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = (20,15)
    if polar_wanted:
        plt.clf()
        plt.ylabel('Imaginary', fontsize=36)
        plt.xlabel('Real', fontsize=36)
        plt.axhline(0, c='black')
        plt.axvline(0, c='black')
        plt.tick_params(axis='both', which='major', labelsize=30)
        scaled1 = scaling(input_vals)
        scaled2 = scaling_same_scaler(input_vals, window_len)
        for x in range(0, len(input_vals)):
            plt.plot([0,scaled1[x].real],[0,scaled1[x].imag],'o-',color='steelblue')
            plt.plot([0,scaled2[x].real],[0,scaled2[x].imag],'o-',color='coral')
        plt.xlim((-1, 1))
        plt.ylim((-1,1))
        plt.legend(['scaling1', 'scaling_same_scaler'], fontsize=36)
        plt.show()
    
    vals = scaling_same_scaler(input_vals, window_len)
    plt.clf()
    plt.figure(figsize=(20,15))
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.title(title, fontsize=36)
    plt.plot(vals.real, color='blue')
    plt.plot(vals.imag, color='coral')
    plt.xlabel('Frequency features', fontsize=36)
    plt.ylabel('Amplitude', fontsize=36)
    plt.legend(['Real', 'Imaginary'], fontsize=36)
    plt.show()

def do_fft(col, window_len, title, plot, BW, real_valued=False):
    """ Calculates FFT for the 1 ms snapRearranges the resulted in 
        tensor with FFTshift.
    Args:
        df (pd.DataFrame): Dataframe from which the snapshot is taken
        window_len (Integer): window_len length for FFT
        key (String): Column name, 
        plot_complex (bool, optional): Tells whether to plot the given results. Defaults to False.
    """
    if real_valued:
        win = hann(window_len, True)
        _, pxx= welch(col, BW, window=win, noverlap=window_len//2, nfft=window_len, scaling='spectrum', return_onesided=False)
        input_val = torch.fft.fftshift(10*torch.log10(torch.tensor(pxx)))
        if plot:
            plt.plot(input_val)
            plt.ylim(-80, 20)
            plt.show()
        scaler = MinMaxScaler(feature_range=(-1,1))
        return scaler.fit_transform(input_val.reshape(-1,1)).flatten()
    else:
        input_val = torch.fft.fftshift(torch.fft.fft(torch.as_tensor(np.array(col)), window_len))
        if plot:
            plot_complex_vectors(input_val, title, window_len)
        return scaling_same_scaler(input_val, window_len).numpy()

def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'requires grad: {name}, {param.size()}')
        else:
            print(f'requires no grad: {name}, {param.size()}')

def plot_prediction_losses(list_of_endings, threshold_max, threshold_min, loss_vals):
    font = {'size': 28}
    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = (27, 20)
    df_to_plot = pd.DataFrame({'losses': loss_vals})
    df_to_plot['index'] = df_to_plot.index
    df_to_plot['category'] = 'Clean'
    START = 0

    for sample in list_of_endings:
        df_to_plot.loc[START:START+sample[1], 'category'] = sample[0]
        print(np.min(loss_vals[START:START+sample[1]]))
        START = sample[1]

    plt.clf()
    plt.title('Loss values for different (CVAuto')
    sns.scatterplot(data=df_to_plot, hue='category', style='category', x='index', y='losses', s=50)
    plt.axhline(y = threshold_max, color = 'red', label = 'Threshold_max')
    plt.axhline(y = threshold_min, color = 'red', label = 'Threshold_min')
    plt.xlabel('Test sample')
    plt.ylabel('Mean Squared Error')
    plt.ylim(0,0.09)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
