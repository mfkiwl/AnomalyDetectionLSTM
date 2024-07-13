import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch, hann

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
            if len(df[key][i:i+len_seq]) == len_seq:
                test_data.append(df[key][i:i+len_seq])
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
    """ Calculates FFT for the snapshot and rearranges the resulted in 
        tensor with FFTshift.
    Args:
        col: data snapshot
        window_len (Integer): window_len length for FFT
        title (String): plot title
        plot (Boolean): Should the values be plotted
        BW (Integer): bandwidth
        real_valued (bool, optional): Tells which operation should be done - complex or real-valued.
    """
    if real_valued:
        win = hann(window_len, True)
        _, pxx= welch(col, BW, window=win, noverlap=window_len//2, nfft=window_len, scaling='spectrum', return_onesided=False)
        input_val = torch.fft.fftshift(10*torch.log10(torch.tensor(pxx)))
        if plot:
            plt.plot(input_val)
            plt.title(title)
            plt.ylim(-50, 20)
            plt.show()
        scaler = MinMaxScaler(feature_range=(-1,1))
        return scaler.fit_transform(input_val.reshape(-1,1)).flatten()
    else:
        input_val = torch.fft.fftshift(torch.fft.fft(torch.as_tensor(np.array(col)), window_len))
        if plot:
            plot_complex_vectors(input_val, title, window_len)
        return scaling_same_scaler(input_val, window_len).numpy()

def write_model_params(model, model_path):
    with open(f'{model_path}_model_parmas.txt', 'w') as model_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                model_params.write(f'requires grad: {name}, {param.size()}')
            else:
                model_params.write(f'requires no grad: {name}, {param.size()}')
