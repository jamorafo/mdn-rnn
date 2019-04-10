
import torch

# Initialize

x = torch.Tensor(2, 3)  # An un-initialized Tensor object. x holds garbage data.
y = torch.rand(2, 3)    # Initialize with random values

# Operations

z1 = x + y
z2 = torch.add(x, y)             # Same as above

print(z2)
print(z1)


#(num_layers * num_directions, batch, hidden_size)
num_layers = 1
hidden_size=4
input_size =5
batch_size=3
seqlen = 2


rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
input = torch.randn(seqlen, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size) #(num_layers * num_directions, batch, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))



