import torch

def generateAdversarialImage(model, input_batch, target_class):
    """
    Generate an adversarial image by applying perturbations to the input image.
    """
    epsilon = 0.001
    num_steps = 3
    alpha = epsilon / num_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.tensor([int(target_class)], device=device)
    input_batch = input_batch.to(device)
    model.to(device)
    input_batch_adv = input_batch.clone()

    for _ in range(num_steps):
        input_batch_adv.requires_grad = True
        output = model(input_batch_adv)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            input_batch_adv += alpha * input_batch_adv.grad.sign()
            input_batch_adv = torch.clamp(input_batch_adv, 0, 1)
        input_batch_adv = input_batch_adv.detach()

    return input_batch_adv
