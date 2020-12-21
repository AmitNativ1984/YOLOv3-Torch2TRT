import torch
from torch2trt import TRTModule

def save_trt_model(model_trt, path):
    """ save model"""
    torch.save(model_trt.state_dict(), path)
    print("tensorrt model serialized and saved to: " + path)

def load_trt_model(path):
    """ load model """
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path))
    print("tensorrt model loaded from: " + path)
    return model_trt