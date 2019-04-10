# # Confidentiel / Proprieté de PolyMtl
# # LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.functions import split_sequences
from src.LSTM import LSTM
################################################################################
# #  Parameters
################################################################################

folder = "20190410"
test_name = "TEST01_LSTM"
variables= ['UWRPM', 'RATE', 'SPDREF', 'UWDIAM_P', 'RRPOS']

#n_obs = 172797
#epochs = 10000
n_obs = 10000
epochs = 1000
batch_size=50

input_size=5
output_size = input_size
seqlen=50
hidden_size = 10
n_layers = 1

################################################################################
# #  Data
################################################################################

rawdata = pd.read_csv('input/R-221a-Oct-04_TR_20181002_00h00.csv', skiprows=1, nrows=n_obs,header=0)
print(rawdata.columns)
data = rawdata[variables]
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.data_min_)

# # Data standardization
data_scaled = scaler.transform(data)
data_scaled0 = Variable(torch.from_numpy(data_scaled))
data_scaled0 = data_scaled0.float()

# # Data setting up. Splitting into sequences.
X, Y = split_sequences(np.array(data_scaled0), seqlen)
X0 = torch.from_numpy(X)
inputs0 = X0.transpose(0,1)

# # Setting up the output according to the delay.
n_batches = inputs0.size(1)
delay=2
index_targets = torch.tensor(np.arange(delay,n_batches))
targets = torch.index_select(inputs0,1,index_targets)
#print(targets)
#print(inputs0[:,-1,:])
#print(targets.shape)
# # Setting up the input according to the delay.
index_inputs = torch.tensor(np.arange(0,n_batches-delay), requires_grad=True)
inputs = torch.index_select(inputs0,1,index_inputs)
#print(inputs)
#print(inputs0[:,-1,:])
#print(inputs.shape)
#print(targets[:,0,:])
#print(inputs[:,0,:])


################################################################################
# #  Functions
################################################################################

model = LSTM(input_size, hidden_size, seqlen, output_size,n_layers)
loss_fn = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.Adam(model.parameters())


################################################################################
# #  Model training
################################################################################

hist = np.zeros(epochs)

for t in range(epochs):
    # Clear stored gradient
    model.zero_grad()

    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()

    # Forward pass
    y_pred = model(inputs)

    loss = loss_fn(y_pred, targets)

    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

print(targets[:,50,:])
print(y_pred[:,50,:])

################################################################################
# #  Visual Evaluation
################################################################################


seq = 1000
true_values = targets[seqlen-1,:,:]
pred_values = y_pred[seqlen-1,:,:]

true_values = true_values.numpy()
pred_values = pred_values.detach().numpy()

t0 = inputs.size(1) - 5000
t1 = inputs.size(1)
for cc in range(0,5):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(true_values[t0:t1,cc], "blue", markersize=2,label='True value')
    plt.plot(pred_values[t0:t1,cc], "red", markersize=2,label='Predicted')
    plt.legend(loc='best', ncol=1, shadow=False, fancybox=False)
    plt.title(variables[cc])
    fig.savefig("output/" + folder + "/" + test_name + "/" + variables[cc] + "n" + str(n_obs) + "e" + str(epochs) + ".pdf",
            bbox_inches='tight')

