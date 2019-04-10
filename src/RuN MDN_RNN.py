# # Confidentiel / Proprieté de PolyMtl
# # Mixture density Network
import torch
import torch.nn as nn
import torch.optim as optim
from src import mdn_RNN
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device
# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]



folder = "20180401"
test_name = "TEST01"
variables= ['UWRPM', 'RATE', 'SPDREF', 'UWDIAM_P', 'RRPOS']

#n_obs = 172797
#epochs = 10000
n_obs = 1000
epochs = 100
batch_size=50

input_size=5
seqlen=6
hidden_size = 5
num_layers = 1

rawdata = pd.read_csv('input/R-221a-Oct-04_TR_20181002_00h00.csv', skiprows=1, nrows=n_obs,header=0)
print(rawdata.columns)
data = rawdata[variables]
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.data_min_)
data_scaled = scaler.transform(data)

#X_train, X_valid = train_test_split(data_scaled, test_size=0.3, random_state=42)
X_train = data_scaled
train_set = Variable(torch.from_numpy(X_train))
train_set = train_set.float()


#valid_set = Variable(torch.from_numpy(X_valid))
#valid_set = valid_set.float()

# initialize the model
model = mdn_RNN.MDN_RNN(input_size, hidden_size, 3, num_layers) #input_size,hidden_size , n_gaussians, num_layers

#optimizer = optim.Adam(model.parameters())
#optimizer = optim.SGD(model.parameters(),lr=0.09, momentum=0.9, nesterov=True)
optimizer = optim.RMSprop(model.parameters(), lr=0.002, alpha=0.999)

loss_train = []
loss_valid = []


rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
X, y = split_sequences(np.array(train_set), seqlen)
#rnn = nn.LSTM(input_size, hidden_size, num_layers)
X, y = split_sequences(np.array(train_set), seqlen)
#(seqlen, batch_size, input_size)
Xinput = torch.from_numpy(X)
inputs = Xinput.transpose(0,1)

hidden = torch.zeros(1, batch_size, 5)
h0 = torch.randn(1, seqlen, 5)
c0 = torch.randn(1, seqlen, 5)

hidden = torch.zeros(1, 994, hidden_size) #of shape (num_layers * num_directions, batch, hidden_size  If the RNN is bidirectional, num_directions should be 2, else it should be 1.
(pi, mu, sigma), hidden = model(inputs, hidden)
output, (hn, cn) = rnn(inputs, (h0, c0))


# Train the model
for epoch in range(epochs):
    # Set initial hidden and cell states
    #hidden = model.init_hidden(batch_size)
    hidden = torch.zeros(1, batch_size, hidden_size) #of shape (num_layers * num_directions, batch, hidden_size  If the RNN is bidirectional, num_directions should be 2, else it should be 1.
    for i in range(0, input_size - seqlen, seqlen):
        # Get mini-batch inputs and targets
        inputs = train_set[i:i + seqlen, :]
        targets = train_set[(i + 1):(i + 1) + seqlen, :]

        # Forward pass
        hidden = detach(hidden)
        (pi, mu, sigma), hidden = model(inputs, hidden)
        loss = mdn_RNN.mdn_loss(pi, sigma, mu, targets)

        loss_train.append(loss.data)
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    #if epoch % 100 == 0:
        #print(epoch, loss.data[0])



fig = plt.figure(figsize=(8, 8))
plt.plot(loss_train, 'b', linewidth=1.0, markersize=2,label='Loss')
#plt.plot(loss_valid, 'r', linewidth=1.0, markersize=2,label='Loss valid')
plt.show()

np.arange(2)
test_set = Variable(torch.from_numpy(data_scaled))
test_set = test_set.float()

pis_test, sigmas_test, mus_test = model(test_set)

values, indices = pis_test.max(1)
n = test_set.shape[0]
mu_test0 = []
sigma_test0 = []
mus_test = mus_test.data.numpy()
sigmas_test = sigmas_test.data.numpy()

for k in range(0,n-1):
    mu_test0.append(mus_test[k,indices.data[0],:])
    sigma_test0.append(sigmas_test[k,indices.data[0],:])

mu_test = np.asarray(mu_test0)
sigma_test = np.asarray(sigma_test0)


upper_limits = mu_test + 1.959964*sigma_test
lower_limits = mu_test - 1.959964*sigma_test


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None,label_line=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, markersize=2,label=label_line)

mu_raw = scaler.inverse_transform(mu_test)
upl_raw = scaler.inverse_transform(upper_limits)
lol_raw = scaler.inverse_transform(lower_limits)

t0 = 1
tn = 10000


for cc in range(0,5):
    fig = plt.figure(figsize=(8, 8))
    plot_mean_and_CI(mu_raw[t0:tn,cc], lol_raw[t0:tn,cc], upl_raw[t0:tn,cc], color_mean="blue", color_shading="blue",label_line='Predicted')
    plt.plot(np.asarray(data)[t0:tn,cc],color='red', markersize=2,label='True value')
    plt.legend(loc='best', ncol=1, shadow=False, fancybox=False)
    plt.title(data.columns[cc])
    fig.savefig("output/"+ folder +"/" + test_name + "/MDN_" + data.columns[cc] + "n"+n_obs +"e" + epochs +".pdf", bbox_inches='tight')

