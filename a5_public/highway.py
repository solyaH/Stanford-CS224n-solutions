#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
### YOUR CODE HERE for part 1h
# from collections import namedtuple
# import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """ Simple HighwayNet """

    def __init__(self, embed_word_size):
        """ Init Highway Model.

        @param embed_word_size (int): Size of the final word embedding (dimensionality)
        """
        super(Highway, self).__init__()
        self.embed_word_size = embed_word_size
        self.c_projection = nn.Linear(embed_word_size, embed_word_size)  # W_proj
        self.g_projection = nn.Linear(embed_word_size, embed_word_size)  # W_gate

    def forward(self, conv_out):
        """
        @param conv_out (Tensor): Tensor of convolutional network output with shape (b, e), where
                                        b = batch_size, e = size of word embedding.
        @returns highway_out(Tensor): Tensor of highway output with shape (b, ).
        """
        conv_out_projection = torch.relu(self.c_projection(conv_out))  # W_proj(c_o) -> (b, e)*(e, e) -> (b, e)
        gate_projection = torch.sigmoid(self.g_projection(conv_out))

        highway_out = gate_projection*conv_out_projection + (1-gate_projection)*conv_out

        return highway_out
