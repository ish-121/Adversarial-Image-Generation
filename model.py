from torchvision import models
def loadModel():
    """
    Load and return the pre-trained model.
    """
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    model.eval()  # Set the model to evaluation mode.
    return model, weights
