import torch
import numpy as np
import glob
import os
import math
from utils.helper_functions import do_fft

def get_data_files(files_to_read):
    print(files_to_read)
    data_files = []
    data_files = glob.glob(files_to_read)
    data_files.sort()
    print("Number of files:", len(data_files))
    return data_files
 
def read_data_from_dat(data_set, files_to_read, window, step_size, BW, sample_interval=20, len_seq=4, real_valued=False):
    data_files = get_data_files(files_to_read)
    for numerator, file_name in enumerate(data_files):
        print(f'File {numerator+1}/{len(data_files)} - ',end=' ')
        name = file_name.replace('.DAT', '').split('/')[-1]
        name = name[-1]
        bin_type = np.int16

        file_size = os.path.getsize(file_name)
        total_ms = int(file_size / BW / 4) * 1000
        plot_sample = True

        after_fft = []
        for j in range(0, int(total_ms), sample_interval):
            # binary file of integers IQIQIQIQ... convert it to complex numbers after which computer FFT
            offset_ = int(math.floor(j * BW * 4/1000))
            if offset_ + step_size * 2 * len_seq > file_size:
                break
            raw_signal = np.fromfile(file_name, dtype=bin_type, offset=offset_, count=(step_size * 2 * len_seq))
            raw_signal = raw_signal.astype(np.float32).view(np.complex64)

            for i in range(0, len_seq):
                after_fft.append(do_fft(raw_signal[i * step_size: (i +1) * step_size], window, plot_sample, BW, real_valued))
                plot_sample = False
        print(f'Number of samples after fft: {len(after_fft)}')
        data_set[name] = torch.tensor(np.array(after_fft))
