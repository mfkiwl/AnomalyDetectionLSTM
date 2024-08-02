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

def read_data_from_dat(data_set, files_to_read, window, BW=int(25e6), sample_length=1000, sample_interval=1e-3, real_valued=False):
    """
        BW: bandwidth
        sample_length: length of the sample in micro seconds, for example 1 ms would be 1000 
        sample_interval: used to determine the number of samples for the specific time interval, for example 1 ms would be 1e-3
        if the bandwidth is xx MHz.
        NOTE: you may want to change the break condition on line 38 in case of long files: if the file is 3 seconds long, with the above definitions you will get 3000 samples/file.
    """
    data_files = get_data_files(files_to_read)
    for numerator, file_name in enumerate(data_files):
        print(f'File {numerator+1}/{len(data_files)} - ',end=' ')
        name = file_name.replace('.DAT', '').replace('.dat', '').split('/')[-1]
        name = name[-1]
        bin_type = np.int16

        file_size = os.path.getsize(file_name)
        total_microsec = int(file_size / BW / 4) *1e6
        plot_sample = True
        num_samples = int(sample_interval * BW)
        after_fft = []
        for j in range(0, int(total_microsec), sample_length):
            offset_ = int(math.floor(j * BW * 4 * 1e-6))
            if offset_ + num_samples * 2 > file_size:
                break
            raw_signal = np.fromfile(file_name, dtype=bin_type, offset=offset_, count=(num_samples * 2))
            raw_signal = raw_signal.astype(np.float32).view(np.complex64)

            after_fft.append(do_fft(raw_signal, window, name, plot_sample, BW, real_valued))
            plot_sample = False
        print(f'Number of samples after fft: {len(after_fft)}')
        data_set[name] = torch.tensor(np.array(after_fft))
