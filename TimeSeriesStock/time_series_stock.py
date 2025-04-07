import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset,DataLoader


def load_data(ticker='AAPL',start_date='2020-01-01',end_date='2025-01-01'):
    data=yf.download(ticker,start=start_date,end=end_date)
    return data[['Close']]

def preprocess_data(data,seq_len=60):
    scaler=MinMaxScaler(feature_range=(-1,1))
    scaled_data=scaler.fit_transform(data.values)

    seqs=[]
    labels=[]
    for i in range(len(scaled_data)-seq_len):
        seqs.append(scaled_data[i:i+seq_len])
        labels.append(scaled_data[i+seq_len])


class TimeSeriesTransformer(nn.Module):
    def __init__(self,input_dim=1,num_heads=4,num_layers=2,dim_feedforward=256):
        super().__init__()
        self.positional_encoding=PositionalEncoding(input_dim)
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            bacth_first=True
        )
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.fc=nn.Linear(input_dim,1)
    