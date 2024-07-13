"""_summary_
@author: outisa
"""
import torch
from torch import nn
import numpy as np
torch.manual_seed(2022)

# Autoencoder
class ComplexAutoencoder(nn.Module):
    def __init__(self, features_n, hidden_dim, num_layers, seq_length) -> None:
        super(ComplexAutoencoder, self).__init__()
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

def initialize_weights(dim1, dim2, criterion='glorot', seed=None):
    """_summary_
    Args:
        dim1 (Integer): Dimension of the input layer
        dim2 (Integer): _description_
        criterion (str, optional): Based on given criterion, the standard 
            deviation value is calculated and used when drawing the modulus from
            the Rayleigh distribution. Defaults to 'glorot'.
        seed (Integer, optional): Seed value. if not given torch.initial_seed is called.
            Defaults to None.

    Raises:
        ValueError: if wrong criterion name is given.
    Code based on authors: Chiheb Trabelsi et all, Paper 'Deep Complex Network' can
        be found on https://arxiv.org/abs/1705.09792
    Returns:
        complex128 tensor: Tensor containing complex valued weights
    """
    std = 0
    if criterion == 'glorot':
        std = 1.0/ np.sqrt(dim1 + dim2)
    elif criterion == 'he':
        std = 1.0/ np.sqrt(dim1)
    else:
        raise ValueError(f'Only "glorot" or "he" are accepted as criterion. {criterion} was given')
    if seed is None:
        seed = torch.initial_seed()
    rs = np.random.RandomState(seed=seed)

    weight_size = (dim2, dim1)

    modulus = rs.rayleigh(scale=std, size=weight_size)
    phase = rs.uniform(low=-np.pi, high=np.pi, size=weight_size)
    weights_real = modulus * np.cos(phase)
    weights_imag = modulus * np.sin(phase)
    return torch.from_numpy( weights_real + 1j * weights_imag).to(torch.cdouble)
