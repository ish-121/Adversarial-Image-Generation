import os
import pandas as pd
from model import loadModel
from image_utils import preprocessImage, saveAdversarialImage
from adversarial_utils import generateAdversarialImagePGD
from classification import classifyImage

def load_class_names(excel_path="classes.xlsx"):
    """
    Load class names from an Excel file and return a dictionary mapping class IDs to class names.
    """
    df = pd.read_excel(excel_path)
    class_names_dict = pd.Series(df['Class Name'].values, index=df['Class ID']).to_dict()
    return class_names_dict

def selectUserImage(folder_path="images"):
    """
    Display available images in a folder and allow the user to select one.
    """
    images = os.listdir(folder_path)
    for i, image in enumerate(images):
        print(f"{i+1}: {image}")
    choice = int(input("Select an image by number: ")) - 1
    return os.path.join(folder_path, images[choice])

import os

def main():
    class_names = load_class_names()  # Load class names mapping
    model, weights = loadModel()
    target_class = input("Enter the target class: ")

    results = []

    # Iterate over all images in the "images" folder
    for image_name in os.listdir("images"):
        image_path = os.path.join("images", image_name)
        input_batch = preprocessImage(image_path, weights)

        # Classify original image
        original_preds = classifyImage(model, input_batch)
        print(f"Original image predictions for {image_name}:")
        for rank, (class_id, confidence) in enumerate(original_preds, start=1):
            class_name = class_names.get(class_id, "Unknown class")
            print(f"Rank {rank}: Class {class_id}, {class_name}, with confidence {confidence*100:.2f}%")

        # Parameter search
        num_steps_range = [10, 20, 30]
        epsilon_range = [0.001, 0.005, 0.01]
        alpha_range = [0.001, 0.003, 0.005]

        for num_steps in num_steps_range:
            for epsilon in epsilon_range:
                for alpha in alpha_range:
                    print(f"\nParameters: num_steps={num_steps}, epsilon={epsilon}, alpha={alpha}")
                    
                    # Generate adversarial image
                    input_batch_adv = generateAdversarialImagePGD(model, input_batch, target_class, epsilon, num_steps, alpha)
                    adv_image_path = saveAdversarialImage(input_batch_adv, image_path, num_steps, epsilon, alpha)

                    # Classify adversarial image
                    adv_preds = classifyImage(model, input_batch_adv)
                    print(f"Adversarial image predictions (saved at {adv_image_path}):")
                    top_classes = []
                    for rank, (class_id, confidence) in enumerate(adv_preds, start=1):
                        class_name = class_names.get(class_id, "Unknown class")
                        print(f"Rank {rank}: Class {class_id}, {class_name}, with confidence {confidence*100:.2f}%")
                        top_classes.append((class_id, class_name, confidence))

                    # Output target class confidence
                    target_class_id = int(target_class)
                    target_class_name = class_names.get(target_class_id, "Unknown class")
                    target_confidence = next((conf for cls_id, conf in adv_preds if cls_id == target_class_id), 0.0)
                    print(f"Target Class {target_class_id}, {target_class_name}, with confidence {target_confidence*100:.2f}%")

                    results.append((image_name, num_steps, epsilon, alpha, top_classes, target_class_id, target_class_name, target_confidence))

    output_results(results)

def output_results(results):
    print("\nFinal Results Table:")
    print("Image Name | Num Steps | Epsilon | Alpha | Top 3 Classes (ID, Name, Confidence) | Target Class (ID, Name, Confidence)")
    print("-" * 120)
    for result in results:
        image_name, num_steps, epsilon, alpha, top_classes, target_class_id, target_class_name, target_confidence = result
        top_classes_str = ", ".join(f"({class_id}, {class_name}, {confidence*100:.2f}%)" for class_id, class_name, confidence in top_classes)
        print(f"{image_name} | {num_steps} | {epsilon} | {alpha} | {top_classes_str} | ({target_class_id}, {target_class_name}, {target_confidence*100:.2f}%)")
if __name__ == "__main__":
    main()
