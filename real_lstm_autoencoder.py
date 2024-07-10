import torch
from  torch import nn

class RealAutoencoder(nn.Module):
    def __init__(self, features_n, hidden_dim, num_layers, seq_lenght) -> None:
        super(RealAutoencoder, self).__init__()
        # expected features, hidden sizes, number of stacked layers, batch_first=True means that
        # input shape should be N,L,H_in -> batch size, sequence length, input size
        self.features_n = features_n
        self.seq_len = seq_lenght
        self.hidden_dim = hidden_dim
    
        self.lstm_in1 = nn.LSTM(self.features_n,self.hidden_dim*2, num_layers, dropout=0.4, batch_first=True, bias=True).to(torch.double)
        self.lstm_in2 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, num_layers, dropout=0.4, batch_first=True, bias=True).to(torch.double)
        self.lstm_out1 = nn.LSTM(self.hidden_dim, self.hidden_dim*2, num_layers, dropout=0.4, batch_first=True, bias=True).to(torch.double)
        self.lstm_out2 = nn.LSTM(self.hidden_dim*2, self.features_n , num_layers, dropout=0.4, batch_first=True, bias=True).to(torch.double)
        self.linear = nn.Linear(self.features_n, self.features_n).to(torch.double)
    
    def forward(self, inputs, hidden=None):
        h, _ = self.lstm_in1(inputs)
        h, _ = self.lstm_in2(h)
        h, _ = self.lstm_out1(h)    
        h, _ = self.lstm_out2(h)
        return self.linear(h)
