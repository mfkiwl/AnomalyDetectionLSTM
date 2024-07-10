"""_summary_

Returns:
    _type_: _description_
"""
import torch
import pandas as pd
import numpy as np
from utils.helper_functions import do_fft
import glob
import matplotlib.pyplot as plt


def get_data_files(files_to_read):
    print(files_to_read)
    data_files = []
    data_files = glob.glob(files_to_read)
    data_files.sort()
    print("Number of files:", len(data_files))
    return data_files
 
def read_data_from_dat(files_to_read, year, window, n_packets):
    pass
