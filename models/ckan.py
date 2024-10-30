# from kan import *
from models.kan import KAN
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class cKAN(nn.Module):
    def __init__(self, num_series, lag, hidden, grid=3, k=3, seed=42):
        '''
        cKAN model with one KAN per time series

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          prun_th: threshold for the prunning.
        '''
        super(cKAN, self).__init__()
        self.p = num_series
        self.lag = lag
        self.hidden = hidden

        # set up KANs
        self.networks = nn.ModuleList([
            KAN(hidden, grid_size=grid, spline_order=k) for _ in range(num_series)
        ])

    def forward(self, x):
        '''
        Forward pass of the cKAN model

        Args:
          x: input tensor of shape (batch_size, num_series, lag)

        Returns:
          y: output tensor of shape (batch_size, num_series)
        '''
        # get the output of each KAN
        y = torch.cat([network(x) for network in self.networks], dim=2)
        return y
    
    def GC(self, array, threshold=None, topk=None):
        '''
        Extract Granger causality matrix from the KANs

        Args:
          threshold: threshold for the prunning.

        Returns:
            GC: Granger causality matrix of shape (num_series, num_series)
        '''
        if topk:
            GC = np.zeros((self.p, self.p))
            for i in range(self.p):
                idx = np.argsort(self.get_scores(array)[i])[::-1][:topk]
                GC[i, idx] = 1
        else:
            GC = torch.zeros(self.p, self.p)
            for i in range(self.p):
                scores = self.networks[i].get_scores().view(self.p, self.lag).sum(dim=1)
                if threshold:
                    GC[i, scores > threshold] = 1
                else:
                    GC[i, scores > 0] = 1

        return GC

    def get_lag_scores(self, array):
        '''
        Get the score of each lag

        Returns:
            output: Score of each lag
        '''
        output = []
        X = rearrange_data(array, self.p, self.lag)

        for i in range(self.p):
            scores = self.networks[i].get_scores(X)
            output.append(scores.detach().numpy())

        output = np.array(output).reshape(self.p, self.p, self.lag)
        return output


    def get_scores(self, array):
        '''
        Get the score of the first layer

        Returns:
            output: Score of each node of the first layer
        '''
        output = []
        X = rearrange_data(array, self.p, self.lag)

        for i in range(self.p):
            scores = self.networks[i].get_scores(X)
            scores = scores.view(self.p, self.lag).sum(dim=1).detach().numpy()
            output.append(scores)

        output = np.array(output)
        return output
    
    
    

def rearrange_data(X, num_series, lag):
    '''
    Rearrange the time series, to form a data with lags

    Input:
        X: tensor with shape (1, num_series, T)
        num_series: number of series
        lag: lag
    Output:
        array: tensor with shape (num_series * lag, T-lag)
    '''
    li = []
    for i in range(num_series):
        for j in range(lag):
            li.append(X[0, :, i].detach().numpy()[j:-lag+j])

    # transfer li to tensor
    array = torch.tensor(np.array(li), dtype=torch.float32).T

    return array

def create_dataset(X, Y, device='cpu'):
    '''
    Create a dataset to satisfy the KAN requirement
    '''
    dataset = {}
    dataset['train_input'] = X.to(device)
    dataset['test_input'] = X.to(device)
    dataset['train_label'] = Y.to(device)
    dataset['test_label'] = Y.to(device)

    return dataset


def train_model_ckan(ckan, array, max_iter=20, lamb=3e-6, device='cpu', lr=1e-4):
    '''
    train the ckan model

    Input:
        ckan: component kan with lots of kan models
        X: time series
        opt: optimizer
        lambd: coefficient for the regularization
        device: used for gpu acceleration
    Output:
        output: the loss of each kan
    '''
    lag = ckan.lag
    num_series = ckan.p
    T = array.shape[1]

    X = rearrange_data(array, num_series, lag)

    train_loss_list = []

    for i in range(num_series):
        Y = array[:, :, i][0, lag:].reshape(1, T-lag).T
        dataset = create_dataset(X, Y, device)
        curr_model = ckan.networks[i]
        optimizer = torch.optim.Adam(ckan.networks[i].parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []
        reg_losses = []

        with trange(max_iter) as t:
            for epoch in t:
                ckan.networks[i].train()
                optimizer.zero_grad()
                output = ckan.networks[i](dataset['train_input'])
                loss = criterion(output, dataset['train_label']) + lamb * ckan.networks[i].regularization_loss(1, 0)
                losses.append(loss.item())
                reg_losses.append(ckan.networks[i].regularization_loss(1, 0).item())
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item(), reg_loss=ckan.networks[i].regularization_loss(1, 0).item())
        train_loss_list.append(losses)


    # for i in range(num_series):
    #     Y = array[:, :, i][0, lag:].reshape(1, T-lag).T
    #     dataset = create_dataset(X, Y, device)
    #     loss_list = ckan.networks[i].fit(dataset, 
    #                                      opt=opt, 
    #                                      steps=max_iter, 
    #                                      lamb=lamb,
    #                                      reg_metric=reg_metric)
    #     train_loss_list.append(loss_list['train_loss'])
        # print(loss_list)

    temp = [np.array(train_loss_list[i]) for i in range(num_series)]
    output = np.array(temp).T

    return output

def plot_GC(GC, GC_estimate):
    '''
    Plot the Granger causality matrix and the estimated Granger causality matrix

    Input:
        GC: Granger causality matrix
        GC_estimate: estimated Granger causality matrix
    '''
    # plot the heatmap of the GC matrix
    fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

    axarr[0].imshow(GC, cmap='Blues')
    axarr[0].set_title('GC actual')
    axarr[0].set_ylabel('Affected series')
    axarr[0].set_xlabel('Causal series')
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    axarr[1].imshow(GC_estimate, cmap='Blues')
    axarr[1].set_title('GC estimated')
    axarr[1].set_ylabel('Affected series')
    axarr[1].set_xlabel('Causal series')
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])

    # Mark disagreements
    for i in range(len(GC)):
        for j in range(len(GC)):
            if GC[i, j] != GC_estimate[i, j]:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
                axarr[1].add_patch(rect)

    plt.show()

def plot_scores(scores):
    '''
    Plot the scores of the first layer

    Input:
        scores: the scores of the first layer
    '''
    # plot the heatmap of the GC matrix
    plt.figure()
    plt.imshow(scores, cmap='Blues')
    plt.title('GC estimated')
    plt.ylabel('Affected series')
    plt.xlabel('Causal series')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_lag_scores(cKAN, X, GC, lag_input, num_series):
    '''
    Plot the scores of each lag
    '''
    for i in range(num_series):
        GC_lag = np.zeros((lag_input, num_series))
        GC_lag[:3, GC[i].astype(bool)] = 1.0

        scores_lag = np.flip(cKAN.get_lag_scores(X)[i].T, axis=0)
        fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
        axarr[0].imshow(GC_lag, cmap='Blues', extent=(0, num_series, lag_input, 0))
        axarr[0].set_title('Series %d true GC' % (i + 1))
        axarr[0].set_ylabel('Lag')
        axarr[0].set_xlabel('Series')
        axarr[0].set_xticks(np.arange(num_series) + 0.5)
        axarr[0].set_xticklabels(range(num_series))
        axarr[0].set_yticks(np.arange(5) + 0.5)
        axarr[0].set_yticklabels(range(1, 5 + 1))
        axarr[0].tick_params(axis='both', length=0)

        axarr[1].imshow(scores_lag, cmap='Blues', extent=(0, num_series, lag_input, 0))
        axarr[1].set_title('Series %d estimated GC' % (i + 1))
        axarr[1].set_ylabel('Lag')
        axarr[1].set_xlabel('Series')
        axarr[1].set_xticks(np.arange(num_series) + 0.5)
        axarr[1].set_xticklabels(range(num_series))
        axarr[1].set_yticks(np.arange(5) + 0.5)
        axarr[1].set_yticklabels(range(1, 5 + 1))
        axarr[1].tick_params(axis='both', length=0)

        plt.show()

