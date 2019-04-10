
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seqlen, output_size,
                    n_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seqlen = seqlen
        self.n_layers = n_layers
        self.output_size = output_size
        #self.hidden = self.init_hidden()

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        # Output layer
        self.linear = nn.Linear(self.hidden_size,  self.output_size)

    def forward(self, input):
        # Forward pass through LSTM layer
        h0 = torch.randn(self.n_layers, self.seqlen, self.hidden_size)  # (num_layers * num_directions, batch, hidden_size)
        c0 = torch.randn(self.n_layers, self.seqlen, self.hidden_size)
        output, _ = self.lstm(input, (h0, c0))
        y_pred = self.linear(output)
        return y_pred




