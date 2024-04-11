import os
from model import loadModel
from image_utils import preprocessImage, saveAdversarialImage
from adversarial_utils import generateAdversarialImagePGD
from classification import classifyImage

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
    input_batch_adv = generateAdversarialImagePGD(model, input_batch, target_class)
    adv_image_path = saveAdversarialImage(input_batch_adv, image_path)

    # Classify adversarial image
    adv_preds = classifyImage(model, input_batch_adv)
    print(f"\nAdversarial image predictions (saved at {adv_image_path}):")
    for rank, (class_id, confidence) in enumerate(adv_preds, start=1):
        print(f"Rank {rank}: Class {class_id} with confidence {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
