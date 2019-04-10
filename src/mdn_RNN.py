"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class MDN_RNN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, input_size,hidden_size , n_gaussians, num_layers):
        super(MDN_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians
        self.num_layers = num_layers

        #self.pi = nn.Sequential(
        #    nn.Linear(input_size, n_gaussians),
        #    nn.Softmax(dim=1)
        #)
        #self.sigma = nn.Linear(input_size, out_features*n_gaussians)
        #self.mu = nn.Linear(in_features, out_features*n_gaussians)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, n_gaussians * input_size)
        self.fc2 = nn.Linear(hidden_size, n_gaussians * input_size)
        self.fc3 = nn.Linear(hidden_size, n_gaussians * input_size)

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)

        pi = pi.view(-1,  self.n_gaussians, self.hidden_size)
        mu = mu.view(-1,  self.n_gaussians, self.hidden_size)
        sigma = sigma.view(-1, self.n_gaussians, self.hidden_size)

        soft_max = nn.Softmax(1)
        pi = soft_max(pi)
        sigma = torch.exp(sigma)
        return pi, mu, sigma


    def forward(self, x,h):
        #pi = self.pi(minibatch)
        #sigma = torch.exp(self.sigma(minibatch))
        #sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        #mu = self.mu(minibatch)
        #mu = mu.view(-1, self.num_gaussians, self.out_features)

        y, (h, c) = self.lstm(x, h)
        #pi, mu, sigma = self.get_mixture_coef(y)
        #return pi, sigma, mu
        #return (pi, mu, sigma), (h, c)
        return (y, (h, c))

    #def init_hidden(self, batch_size):
      #  return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
          #      torch.zeros(self.num_layers, batch_size, self.hidden_size))


def gaussian_probability(sigma, mu, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    data = data.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, data):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, data)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample

