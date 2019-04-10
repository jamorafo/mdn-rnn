# # Confidentiel / Proprieté de PolyMtl
# # Mixture density Network
import torch
import torch.nn as nn
import torch.optim as optim
from src import mdn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

folder = "20181214"
test_name = "TEST01"
variables= ['UWRPM', 'RATE', 'SPDREF', 'UWDIAM_P', 'RRPOS']

#n_obs = 172797
#epochs = 10000
n_obs = 40000
epochs = 10000

rawdata = pd.read_csv('input/R-221a-Oct-04_TR_20181002_00h00.csv', skiprows=1, nrows=n_obs,header=0)
print(rawdata.columns)
data = rawdata[variables]
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.data_min_)
data_scaled = scaler.transform(data)

X_train, X_valid = train_test_split(data_scaled, test_size=0.3, random_state=42)

train_set = Variable(torch.from_numpy(X_train))
train_set = train_set.float()


valid_set = Variable(torch.from_numpy(X_valid))
valid_set = valid_set.float()

# initialize the model
model = nn.Sequential(
    nn.Linear(5,5),
    nn.Tanh(),
    mdn.MDN(5, 5, 3)
)
#optimizer = optim.Adam(model.parameters())
#optimizer = optim.SGD(model.parameters(),lr=0.09, momentum=0.9, nesterov=True)
optimizer = optim.RMSprop(model.parameters(), lr=0.002, alpha=0.999)

loss_train = []
loss_valid = []

for e in range(0,epochs):
    model.zero_grad()
    pi, sigma, mu = model(train_set)
    pi_v, sigma_v, mu_v = model(valid_set)
    loss = mdn.mdn_loss(pi, sigma, mu, train_set)
    loss_v = mdn.mdn_loss(pi_v, sigma_v, mu_v, valid_set)
    if e % 100 == 0:
        print(e, loss.data[0],loss_v.data[0])
    loss_train.append(loss.data)
    loss_valid.append(loss_v.data)
    loss.backward()
    optimizer.step()

fig = plt.figure(figsize=(8, 8))
plt.plot(loss_train, 'b', linewidth=1.0, markersize=2,label='Loss')
plt.plot(loss_valid, 'r', linewidth=1.0, markersize=2,label='Loss valid')
plt.show()


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

