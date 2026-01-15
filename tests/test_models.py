import torch
import pytest
from src.models.pre_fire.lstm_model import FireRiskLSTM
from src.models.post_fire.unet_model import UNet

def test_unet_output_shape():
    model = UNet(in_channels=12, out_channels=1)
    # Batch size 1, 12 channels, 64x64 image
    input_tensor = torch.randn(1, 12, 64, 64)
    output = model(input_tensor)
    assert output.shape == (1, 1, 64, 64), f"Expected (1, 1, 64, 64) but got {output.shape}"

def test_lstm_forward():
    # input_size=12 features, hidden=64
    model = FireRiskLSTM(input_size=12, hidden_size=64, num_layers=2, output_size=1)
    # Batch 1, Sequence length 7 days, 12 features
    input_tensor = torch.randn(1, 7, 12) 
    output = model(input_tensor)
    assert output.shape == (1, 1), f"Expected (1, 1) but got {output.shape}"
