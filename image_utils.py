import os
from PIL import Image
from torchvision import transforms

def preprocessImage(image_path, model_weights):
    """
    Preprocess the selected image according to the model's requirements.
    """
    preprocess = model_weights.transforms()
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Convert to a mini-batch as expected by the model.
    return input_batch

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
