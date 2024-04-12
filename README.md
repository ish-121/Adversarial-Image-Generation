# Adversarial-Image-Generation

# 6. Results and Discussion

The full results can be viewed in the respective “results” Excel files, where there is one for the VGG-13 model, and one for the vit_b_16 model.

Below are tables summarizing the results for both models, where the result for any target class (e.g., 282 - tiger cat) is broadly representative of the other results.

## Table VGG-13 for 282 - tiger cat target class

| Image | Original Top Prediction  | Original Top Confidence | Adversarial Top Prediction | Adversarial Top Confidence |
|-------|--------------------------|-------------------------|----------------------------|----------------------------|
| dog1  | German shepherd          | 0.955                   | goldfish                   | 0.650                      |
| dog2  | Pug                      | 0.910                   | ibex, Capra ibex           | 0.112                      |
| dog3  | Staffordshire terrier    | 0.875                   | bloodhound, sleuthhound    | 0.180                      |
| dog4  | Chihuahua                | 0.951                   | kelpie                     | 0.719                      |
