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

folder = "20190515"
test_name = "TEST01_LSTM"
variables= ['UWRPM', 'RATE', 'SPDREF', 'UWDIAM_P', 'RRPOS']

#n_obs = 172797
#epochs = 10000
n_obs = 100
epochs = 100
batch_size=50
# Delay should be set at 1 to predict the next time step.
delay=1
# time_steps is 1 by default in order to make. Each time step of the test dataset will be walked one at a time.
# A model will be used to make a forecast for the time step, then the actual expected value for the next month
# from the test set will be taken and made available to the model for the forecast on the next time step.
time_steps = 1

input_size=5
output_size = input_size
seqlen=50
hidden_size = 10
n_layers = 1
learning_rate = 0.001

n_sequences = n_obs-seqlen - 1
n_train = round(n_sequences*0.7)
n_valid = round(n_sequences*0.15)
n_test = n_sequences - n_train - n_valid


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

index_targets = torch.tensor(np.arange(delay,n_batches))
targets = torch.index_select(inputs0,1,index_targets)

# # Setting up the input according to the delay.
index_inputs = torch.tensor(np.arange(0,n_batches-delay), requires_grad=True)
inputs = torch.index_select(inputs0,1,index_inputs)


x_train = inputs[:,0:n_train,:]
x_valid = inputs[:,n_train:(n_train+n_valid),:]
x_test = inputs[:,-n_test:,:]

y_train = targets[:,0:n_train,:]
y_valid = targets[:,n_train:(n_train+n_valid),:]
y_test = targets[:,-n_test:,:]


################################################################################
# #  Functions
################################################################################

model = LSTM(input_size, hidden_size, seqlen, output_size,n_layers)
loss_fn = torch.nn.MSELoss(size_average=True)
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)


################################################################################
# #  Model training
################################################################################

losses = np.zeros(shape = (epochs, 2))

for t in range(epochs):
    # Clear stored gradient
    #model.zero_grad()

    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # model.hidden = model.init_hidden()

    # Forward pass
    out_train = model(x_train)
    # we keep only the last output cell to train the LSTM.
    y_pred_train = out_train[-1, :, :]

    loss_train = loss_fn(y_pred_train, y_train[-1,:,:])
    # validation
    out_valid = model(x_valid)
    y_pred_valid = out_valid[-1, :, :]

    loss_valid = loss_fn(y_pred_valid, y_valid[-1, :, :])

    if t % 100 == 0:
        print("Epoch ", t, "MSE train: ", loss_train.item(), "MSE: valid", loss_valid.item())
    losses[t,0] = loss_train.item()
    losses[t,1] = loss_valid.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss_train.backward()

    # Update parameters
    optimiser.step()

#print(y_train[-1,:,:])
#print(y_pred_valid)

#t = epochs-1
#PATH =  "output/" + folder + "/" + test_name + "/" + "n" + str(n_obs) + "e" + str(t)+"lr"+str(learning_rate)+ ".pt"
#torch.save(model.state_dict(), PATH )

#model.load_state_dict(torch.load(PATH))
#model.eval()


################################################################################
# #  Visual Evaluation
################################################################################


x_valid = inputs[:,0:(n_train+n_valid),:]
y_valid = targets[:,0:(n_train+n_valid),:]
#np.random.seed(1234)
out_valid = model(x_valid)

y_pred_valid = out_valid[-1, :, :]

true_values = y_valid[-1,:,:]
pred_values = y_pred_valid

y_pred_valid
true_values



true_values = true_values.numpy()
pred_values = pred_values.detach().numpy()

t0 = 50
t1 = true_values.shape[0]
for cc in range(0,5):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(true_values[t0:t1,cc], "blue", markersize=2,label='True value')
    plt.plot(pred_values[t0:t1,cc], "red", markersize=2,label='Predicted')
    plt.legend(loc='best', ncol=1, shadow=False, fancybox=False)
    plt.title(variables[cc])





fig = plt.figure(figsize=(8, 8))
plt.plot(losses[:,0], "orange", markersize=2,label='Training loss')
plt.plot(losses[:,1], "green", markersize=2,label='Validation loss')
plt.legend(loc='best', ncol=1, shadow=False, fancybox=False)
fig.savefig("output/" + folder + "/" + test_name + "/" + "learning_curves" + "n" + str(n_obs) + "e" + str(epochs) + ".pdf",
        bbox_inches='tight')



x_valid = inputs[:,(n_train-100):(n_train+n_valid),:]
y_valid = targets[:,(n_train-100):(n_train+n_valid),:]

x_valid = inputs[:,0:(n_train+n_valid),:]
y_valid = targets[:,0:(n_train+n_valid),:]
np.random.seed(1234)
out_valid = model(x_valid)

y_pred_valid = out_valid[-1, :, :]

true_values = y_valid[-1,:,:]
pred_values = y_pred_valid

y_pred_valid
true_values



true_values = true_values.numpy()
pred_values = pred_values.detach().numpy()

t0 = 50
t1 = true_values.shape[0]
for cc in range(0,5):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(true_values[t0:t1,cc], "blue", markersize=2,label='True value')
    plt.plot(pred_values[t0:t1,cc], "red", markersize=2,label='Predicted')
    plt.legend(loc='best', ncol=1, shadow=False, fancybox=False)
    plt.title(variables[cc])
    fig.savefig("output/" + folder + "/" + test_name + "/" + variables[cc] + "n" + str(n_obs) + "e" + str(epochs) + ".pdf",
            bbox_inches='tight')



