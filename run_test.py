import os
import torch
from torchvision import models, transforms
from PIL import Image

def loadModel():
    """
    Load and return the pre-trained model.
    """
    weights = models.VGG13_Weights.DEFAULT
    model = models.vgg13(weights=weights)
    model.eval()  # Set the model to evaluation mode.
    return model, weights

def preprocessImage(image_path, model_weights):
    """
    Preprocess the selected image according to the model's requirements.
    """
    preprocess = model_weights.transforms()
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Convert to a mini-batch as expected by the model.
    return input_batch

def generateAdversarialImage(model, input_batch, target_class):
    """
    Generate an adversarial image by applying perturbations to the input image.
    """
    epsilon = 0.005  # Smaller step size for more precise perturbation.
    num_steps = 20  # Number of gradient ascent steps.
    alpha = epsilon / num_steps  # Step size for each iteration.
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

def classifyImage(model, input_batch):
    """
    Classify the image and return the top 3 predictions.
    """
    output = model(input_batch)
    predictions = torch.nn.functional.softmax(output, dim=1)
    top_pred = predictions.topk(3)
    return [(top_pred.indices[0][i].item(), top_pred.values[0][i].item()) for i in range(top_pred.indices.size(1))]

def saveAdversarialImage(input_batch_adv, original_image_path):
    """
    Save the adversarial image to the filesystem.
    """
    adv_image = transforms.ToPILImage()(input_batch_adv.squeeze())
    original_image_name = os.path.basename(original_image_path)
    adv_image_name = "adv_" + original_image_name
    adv_image_path = os.path.join("adv_images", adv_image_name)
    adv_image.save(adv_image_path)
    return adv_image_path

def selectUserImage(folder_path="images"):
    """
    Display available images in a folder and allow the user to select one.
    """
    images = os.listdir(folder_path)
    for i, image in enumerate(images):
        print(f"{i+1}: {image}")
    choice = int(input("Select an image by number: ")) - 1
    return os.path.join(folder_path, images[choice])

def main():
    """
    Main function to execute the adversarial image generation process.
    """
    model, weights = loadModel()
    image_path = selectUserImage()
    target_class = input("Enter the target class: ")
    input_batch = preprocessImage(image_path, weights)

    # Classify original image
    original_preds = classifyImage(model, input_batch)
    print("Original image predictions:")
    for rank, (class_id, confidence) in enumerate(original_preds, start=1):
        print(f"Rank {rank}: Class {class_id} with confidence {confidence*100:.2f}%")

    # Generate adversarial image
    input_batch_adv = generateAdversarialImage(model, input_batch, target_class)
    adv_image_path = saveAdversarialImage(input_batch_adv, image_path)

    # Classify adversarial image
    adv_preds = classifyImage(model, input_batch_adv)
    print(f"\nAdversarial image predictions (saved at {adv_image_path}):")
    for rank, (class_id, confidence) in enumerate(adv_preds, start=1):
        print(f"Rank {rank}: Class {class_id} with confidence {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
