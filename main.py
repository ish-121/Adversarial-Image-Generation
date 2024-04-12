import os
import pandas as pd
from model import loadModel
from utils.image_utils import preprocessImage, saveAdversarialImage
from utils.adversarial_utils import generateAdversarialImagePGD
from utils.classification import classifyImage

def load_class_names(excel_path="utils/classes.xlsx"):
    """
    Load class names from an Excel file and return a dictionary mapping class IDs to class names.
    """
    df = pd.read_excel(excel_path)
    class_names_dict = pd.Series(df['Class Name'].values, index=df['Class ID']).to_dict()
    return class_names_dict

def collect_results(image_name, target_class, class_names, original_preds, adv_preds, adv_image_path):
    """
    Collates all the results together to be in a more report-ready format
    """
    # Get the target class name
    target_class_name = class_names.get(target_class, "Unknown class")
    
    # Get the top 3 predictions for the original image
    original_top3 = [(class_names.get(class_id, "Unknown class"), confidence) for class_id, confidence in original_preds[:3]]
    
    # Get the top 3 predictions for the adversarial image
    adv_top3 = [(class_names.get(class_id, "Unknown class"), confidence) for class_id, confidence in adv_preds[:3]]
    
    # Get the confidence of the target class in the adversarial image
    target_confidence = next((conf for cls_id, conf in adv_preds if cls_id == target_class), 0.0)
    
    # Create a dictionary to store the result
    result = {
        "Image": image_name,
        "Target Class": target_class,
        "Target Class Name": target_class_name,
        "Original Prediction 1": original_top3[0][0],
        "Original Confidence 1": original_top3[0][1],
        "Original Prediction 2": original_top3[1][0],
        "Original Confidence 2": original_top3[1][1],
        "Original Prediction 3": original_top3[2][0],
        "Original Confidence 3": original_top3[2][1],
        "Adversarial Prediction 1": adv_top3[0][0],
        "Adversarial Confidence 1": adv_top3[0][1],
        "Adversarial Prediction 2": adv_top3[1][0],
        "Adversarial Confidence 2": adv_top3[1][1],
        "Adversarial Prediction 3": adv_top3[2][0],
        "Adversarial Confidence 3": adv_top3[2][1],
        "Adversarial Target Class Confidence": target_confidence,
        "Adversarial Image Path": adv_image_path
    }
    
    return result

def main():
    class_names = load_class_names()  # Load class names mapping
    model, weights = loadModel()
    
    # Set the parameters for generating adversarial images
    num_steps = 20
    epsilon = 0.001
    alpha = 0.001
    
    # Define the target classes as 282 - tiger cat, 283 - Persian cat, 284 - Siamese cat, Siamese, 285 - Egyptian cat
    target_classes = [282, 283, 284, 285]
    
    # Create an empty list to store the results
    results = []

    # Iterate over all images in the "images" folder
    for image_name in os.listdir("images"):
        image_path = os.path.join("images", image_name)
        input_batch = preprocessImage(image_path, weights)
        
        # Classify original image
        original_preds = classifyImage(model, input_batch)
        # Iterate over each target class
        for target_class in target_classes:
            # Generate adversarial image
            input_batch_adv = generateAdversarialImagePGD(model, input_batch, target_class, epsilon, num_steps, alpha)
            adv_image_path = saveAdversarialImage(input_batch_adv, image_path, target_class)
            
            # Classify adversarial image
            adv_preds = classifyImage(model, input_batch_adv)
            
            # Collect results
            result = collect_results(image_name, target_class, class_names, original_preds, adv_preds, adv_image_path)
            results.append(result)
    
    # Convert the results to a pandas DataFrame
    df_results = pd.DataFrame(results, columns=[
        "Image", "Target Class", "Target Class Name",
        "Original Prediction 1", "Original Confidence 1",
        "Original Prediction 2", "Original Confidence 2",
        "Original Prediction 3", "Original Confidence 3",
        "Adversarial Prediction 1", "Adversarial Confidence 1",
        "Adversarial Prediction 2", "Adversarial Confidence 2",
        "Adversarial Prediction 3", "Adversarial Confidence 3",
        "Adversarial Target Class Confidence", "Adversarial Image Path"])
    # Save the results to an Excel file
    df_results.to_excel("results.xlsx", index=False)
    
    print("Results saved to 'results.xlsx'.")

#Code for the hyperparameter search
#def output_results(results):
    #print("\nFinal Results Table:")
    #print("Image Name | Num Steps | Epsilon | Alpha | Top 3 Classes (ID, Name, Confidence) | Target Class (ID, Name, Confidence)")
    #print("-" * 120)
    #for result in results:
        #image_name, num_steps, epsilon, alpha, top_classes, target_class_id, target_class_name, target_confidence = result
        #top_classes_str = ", ".join(f"({class_id}, {class_name}, {confidence*100:.2f}%)" for class_id, class_name, confidence in top_classes)
        #print(f"{image_name} | {num_steps} | {epsilon} | {alpha} | {top_classes_str} | ({target_class_id}, {target_class_name}, {target_confidence*100:.2f}%)")

if __name__ == "__main__":
    main()
