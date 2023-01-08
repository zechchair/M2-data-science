"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    """Simple GNN model"""

    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        # Tasks 10 and 13

        ##################

        first_mul = self.fc1(x_in)
        second_mul = torch.mm(adj, first_mul)
        Z_0 = self.relu(second_mul)
        Z_0_droped = self.dropout(Z_0)

        first_mul = self.fc2(Z_0_droped)
        second_mul = torch.mm(adj, first_mul)
        Z_1 = self.relu(second_mul)
        x=self.fc3(Z_1)
        ##################
        
        
        # return F.log_softmax(x, dim=1) # FOR  Tasks 10

        return F.log_softmax(x, dim=1),Z_1
