import torch

def generateAdversarialImagePGD(model, input_batch, target_class, epsilon=0.001, num_steps=20, alpha=0.001):
    """
    Generate an adversarial image by applying perturbations to the input image using Projected Gradient Descent (PGD) method.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.tensor([int(target_class)], device=device)
    input_batch = input_batch.to(device)
    model.to(device)
    input_batch_adv = input_batch.clone()
    
    # Random start within the epsilon ball
    input_batch_adv += (2.0 * epsilon * torch.rand_like(input_batch_adv) - epsilon)
    input_batch_adv = torch.clamp(input_batch_adv, 0, 1)  # Ensure it's still a valid image

    for _ in range(num_steps):
        input_batch_adv.requires_grad = True
        
        output = model(input_batch_adv)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Instead of just taking the sign, we'll scale the gradient by alpha and apply the update
            input_batch_adv += alpha * input_batch_adv.grad.sign()
            
            # Ensure the perturbations are within the epsilon-ball around the original image
            perturbation = torch.clamp(input_batch_adv - input_batch, -epsilon, epsilon)
            input_batch_adv = torch.clamp(input_batch + perturbation, 0, 1)
        
        input_batch_adv = input_batch_adv.detach()

    return input_batch_adv
