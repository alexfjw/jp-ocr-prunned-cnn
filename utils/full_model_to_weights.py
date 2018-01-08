import torch

def model_to_weights(source, dest):
    x = torch.load(source)
    torch.save(x.state_dict(), dest)


