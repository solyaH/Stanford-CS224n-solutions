#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
# from collections import namedtuple
# import sys
from typing import List
import torch
import torch.nn as nn
# import torch.nn.utils
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class CNN(nn.Module):
    """ Simple HighwayNet """

    def __init__(self, embed_char_size, max_word_length, filters_size, kernel_size=5):
        """ Init Highway Model.

        @param embed_size (int): Size of character embedding (dimensionality)
        @param max_word_length (int): Maximum word length
        @param kernel_size (int): Size of the window used to compute features (dimensionality)
        @param filters_size (int): Number of output filters  (dimensionality)
        """
        super(CNN, self).__init__()
        self.embed_char_size = embed_char_size
        self.filters_size = filters_size
        self.kernel_size = kernel_size
        self.max_word_length = max_word_length

        self.conv1d = nn.Conv1d(in_channels=embed_char_size, out_channels=filters_size, kernel_size=kernel_size)
        self.pooling = nn.MaxPool1d(kernel_size=max_word_length - kernel_size + 1)

    def forward(self, x_reshaped):
        """
        @param x_reshaped(Tensor): Tensor of (b, embed_char_size, max_word_length)
        @returns conv_out(Tensor): Tensor of cnn output with shape (b, embed_word_size), where
                                   embed_word_size = size of the final word embedding
        """
        conv = self.conv1d(x_reshaped)
        conv_out = self.pooling(F.relu(conv)).squeeze(-1)

        return conv_out
