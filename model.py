from torchvision import models
def loadModel():
    """
    Load and return the pre-trained model.
    """
    weights = models.VGG13_Weights.DEFAULT
    model = models.vgg13(weights=weights)
    model.eval()  # Set the model to evaluation mode.
    return model, weights
