import torch

def save_model(model, path="meta_model.pth"):
    """ Saves the trained model to a file """
    torch.save(model.state_dict(), path)

def load_model(model_class, path="meta_model.pth"):
    """ Loads a trained model from a file """
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
