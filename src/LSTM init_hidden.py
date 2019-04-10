class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_size,
                    n_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.hidden = self.init_hidden()

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,self.n_layers)
        # Output layer
        self.linear = nn.Linear(self.hidden_size,  self.output_size)

    def init_hidden(self):
        # This is what we'll initialise the hidden state as
        return (torch.zeros(self.n_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.n_layers, self.batch_size, self.hidden_size))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        output, self.hidden = self.lstm(input, self.hidden)
        y_pred = self.linear(output)
        return y_pred