import torch

def classifyImage(model, input_batch):
    """
    Classify the image and return the top 3 predictions.
    """
    output = model(input_batch)
    predictions = torch.nn.functional.softmax(output, dim=1)
    top_pred = predictions.topk(3)
    return [(top_pred.indices[0][i].item(), top_pred.values[0][i].item()) for i in range(top_pred.indices.size(1))]
