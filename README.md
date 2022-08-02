# Plant Disease Detection ☘

This model tries to find out if there is a disease in the crop or not. It takes an RGB image of the plant’ leaf and determines if this plant is healthy or has a disease, and if it has a disease, it determines which kind of disease is in order to help the farmer to find a suitable treatment. For example, if you enter an apple leaf image, it will tell you if it is a healthy apple or has a disease like ‘apple scape’ or ‘black rot'. This is done by extracting some features from the leaf image and then classifying the disease based on these features. 
<br>
<br>
The following figure describes the input and output of the disease detection model:
<br>
<br>
![alt](https://drive.google.com/uc?export=view&id=1ZifwqTNsCiU2eVKTnXZCOVmcz1elrG0X)
<br>
<br>

## Model training
In the dataset called "Plant village dataset" we have almost nine different types of fruits and vegetables, and each type has different kinds of diseases corresponding to it. I train a Random Forest model for each type of plant so it can classify the disease based on the plant type.
<br>

### Feature extraction
The features that are used are: 

- Local binary pattern(LPB) to encode local texture information from the image
- Hu moments which help in describing the outline of a particular leaf. Hu moments are calculated over a single channel only, so the first step is converting the RGB image to Gray scale image and then the Hu moments are calculated.
- Harlick texture to differentiate between healthy leaves and diseased leaves as both of them have different textures.
- Gabor filter which used for texture analysis, which means that it analyzes whether there is any specific frequency content in the image in specific directions in a localized region
- Color Histogram
<br>

The following figure shwos the steps of model training:
<br>
<br>
![alt](https://drive.google.com/uc?export=view&id=1Uxc5M8tssRRjqXTLA0_77PfJq_anei41)
<br>

## Testing Results
I divide the dataset into 60%-20%-20% train, validation and test. So I test my model on the 20% partition and here is the testing results for each plant type model
<br>
<br>
![alt](https://drive.google.com/uc?export=view&id=1EdDZVWk46c3gEo85JzE7nceSYcoSENVh)




