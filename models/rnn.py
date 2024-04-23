#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:52:13 2022

@author: sen
"""
import torch
from torch import nn
import torch.nn.init as init
import numpy as np
from cells.rnncells import RnnCell, GruCell, LstmCell, UrLstmCell
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from cells.hippocells import GatedHippoCell

class SimpleRNN(nn.Module):
    """
    Simple RNN model implementation.

    Args:
        input_size (int): Size of the input feature.
        hidden_size (int): Size of the hidden state.
        output_size (int): Size of the output.
        activation (str, optional): Activation function for the RNN cell. Can be 'tanh', 'relu', or 'sigmoid'. Defaults to 'relu'.
    """
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = RnnCell(input_size=input_size, hidden_size=hidden_size, activation=activation)
        self.h2o = nn.Linear(self.hidden_size, output_size)

        self.initialize_weights()

    def initialize_weights(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for layer in [self.rnn_cell, self.h2o]:
            for param_name, param in layer.named_parameters():
                if 'weight' in param_name:
                    init.uniform_(param, -std, std)
                elif 'bias' in param_name:
                    init.zeros_(param)

    def forward(self, xs, h_t=None):
        """
        Forward pass of the Simple RNN.

        Args:
            xs (torch.Tensor): Input tensor of shape [batchsize, seq_len, 1].
            h_t (torch.Tensor, optional): Initial hidden state tensor. If None, a tensor of zeros will be used. Defaults to None.

        Returns:
            tuple: A tuple containing the output tensor of shape [batchsize, output_size], and the final hidden state tensor of shape [batchsize, hidden_size].
        """
        if h_t is None:
            h_t = torch.zeros(xs.shape[0], self.hidden_size).to(device)

        for t in range(xs.size(1)):
            h_t, _ = self.rnn_cell(x=xs[:, t, :], carry=(h_t, h_t))
        out = self.h2o(h_t)
        return out

class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GruRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_cell = GruCell(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

        self.initialize_weights()

    def initialize_weights(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for layer in [self.rnn_cell, self.h2o]:
            for param_name, param in layer.named_parameters():
                if 'weight' in param_name:
                    init.uniform_(param, -std, std)
                elif 'bias' in param_name:
                    init.zeros_(param)

    def forward(self, xs, h_t=None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, seqence length, inputsize]
        Output: output (torch tensor) of shape [batchsize, outputsize]
        '''

        if h_t is None:
            h_t = torch.zeros(xs.shape[0], self.hidden_size).to(device)

        for t in range(xs.size(1)):
            h_t, _ = self.rnn_cell(xs[:, t, :], carry = (h_t, h_t))

        out = self.h2o(h_t)
        return out

class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super(LstmRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_cell = LstmCell(self.input_size, self.hidden_size, activation = activation)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for layer in [self.rnn_cell, self.h2o]:
            for param_name, param in layer.named_parameters():
                if 'weight' in param_name:
                    init.uniform_(param, -std, std)
                elif 'bias' in param_name:
                    init.zeros_(param)

    def forward(self, xs, h_t = None, c_t = None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, sequence length, inputsize]
        Output: output (torch tensor) of shape [batchsize, outputsize]
        '''

        if h_t is None:
            h_t = torch.zeros(xs.shape[0], self.hidden_size).to(device)
        
        if c_t is None:
            c_t = torch.zeros(xs.shape[0], self.hidden_size).to(device)
                
        for t in range(xs.size(1)):
            h_t, c_t = self.rnn_cell(xs[:, t, :], carry=(h_t, c_t))

        out = self.h2o(h_t)

        return out

class UrLstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(UrLstmRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_cell = UrLstmCell(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

        self.initialize_weights()

    def initialize_weights(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for layer in [self.rnn_cell, self.h2o]:
            for param_name, param in layer.named_parameters():
                if 'weight' in param_name and 'forget_bias' not in param_name:
                    init.uniform_(param, -std, std)
                elif 'bias' in param_name and 'forget_bias' not in param_name:
                    init.zeros_(param)


    def forward(self, xs, h_t = None, c_t = None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, sequence length, inputsize]
        Output: output (torch tensor) of shape [batchsize, outputsize]
        '''

        if h_t is None:
            h_t = torch.zeros(xs.shape[0], self.hidden_size).to(device)
        
        if c_t is None:
            c_t = torch.zeros(xs.shape[0], self.hidden_size).to(device)
            
        for t in range(xs.size(1)):
            h_t, c_t = self.rnn_cell(xs[:, t, :], carry=(h_t, c_t))

        out = self.h2o(h_t)

        return out

class HippoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HippoRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cell = GatedHippoCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 10)

        self.initialize_weights()

    def initialize_weights(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for layer_name, layer in self.named_children():
            if layer_name in ['i2h', 'h2h']:
                for param_name, param in layer.named_parameters():
                    if 'weight' in param_name:
                        init.uniform_(param, -std, std)
                    elif 'bias' in param_name:
                        init.zeros_(param)

    def forward(self, xs, carry=(None, None)):
        if carry[0] is None:
            carry = (
                torch.zeros(xs.shape[0], self.hidden_size).to(device),
                torch.zeros(xs.shape[0], self.hidden_size).to(device),
            )

        for t in range(xs.size(1)):

            # shape xs :=: (batchsize, 28*28, 1)
            # shape xs[:, t, :] :=: (batchsize, 1)

            # Input to hippo cell should be :-: shape: f_t :-: (batchsize, 1, 1)
            # Carry to hippo cell should be :-: shape: c_t :-: (batchsize, 1, N)
            # Carry to hippo cell should be :-: shape: h_t :-: (batchsize, 1, N)
            carry = self.cell(x=xs[:, t, :], carry=carry, t=t)
            c_t = carry[1]
        return self.fc(c_t)

def test_rnn():
  # batch size, sequence length, input size
    model = SimpleRNN(input_size=1, hidden_size=128, output_size=10)
    model = model.to(device)
    x = torch.randn(64, 28*28, 1).to(device)
    out = model(x)
    xshape = out.shape
    return x, xshape

def test_gru():
  # batch size, sequence length, input size
    model = GruRNN(input_size=1, hidden_size=128, output_size=10).to(device)
    model = model.to(device)
    x = torch.randn(64, 28*28, 1).to(device)
    out = model(x)
    xshape = out.shape
    return x, xshape


def test_lstm():
  # batch size, sequence length, input size
    model = LstmRNN(input_size=1, hidden_size=128, output_size=10).to(device)
    model = model.to(device)
    x = torch.randn(64, 28*28, 1).to(device)
    out = model(x)
    xshape = out.shape
    return x, xshape


def test_urlstm():
  # batch size, sequence length, input size
    model = UrLstmRNN(input_size=1, hidden_size=128, output_size=10).to(device)
    model = model.to(device)
    x = torch.randn(64, 28*28, 1).to(device)
    out = model(x)
    xshape = out.shape
    return x, xshape

#testx, xdims = test_rnn()
#print("Simple RNN size test: passed.")

#testx, xdims = test_gru()
#print("Gru RNN size test: passed.")\

#testx, xdims = test_lstm()
#print("LSTM RNN size test: passed.")

#testx, xdims = test_urlstm()
#print("UR LSTM RNN size test: passed.")

