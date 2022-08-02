# %%
from skimage.util import img_as_ubyte
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import cv2
from skimage import io
from skimage.filters import threshold_otsu
from skimage.segmentation import  flood_fill
from skimage.feature import peak_local_max
from skimage.feature import hog
import math
from matplotlib.pyplot import bar
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import mahotas as mt
from sklearn.metrics import accuracy_score 
from skimage.feature import local_binary_pattern
from sklearn import preprocessing

# %%
dictionary = {
    # apple
    0: 'Apple scab',
    1: 'Black rot',
    2: 'apple rust',
    3: 'Apple is healthy',

    # cherry
    5: 'Cherry is healthy',
    6: 'Powdery mildew',

    # corn
    7: 'Gray leaf spot',
    8: 'Common rust',
    9: 'Corn is healthy',
    10: 'Northern Leaf Blight',

    # grape
    11: 'Black rot',
    12: 'Black Measles',
    13: 'Grape is healthy',
    14: 'Leaf blight',

    # peach
    16: 'Bacterial spot',
    17: 'Peach is healthy',

    # pepper
    18: 'Bacterial spot',
    19: 'Pepper is healthy',

    # potato
    20: 'Early blight',
    21: 'Potato is healthy',
    22: 'Late blight',

    # Strawberry
    26: 'Strawberry is healthy',
    27: 'Leaf scorch',

    # Tomato
    28: 'Bacterial spot',
    29: 'Early blight',
    30: 'Tomato is healthy',
    31: 'Late blight',
    32: 'Leaf Moldy',
    33: 'Septoria leaf spot',
    34: 'Two-spotted spider mite',
    35: 'Target Spot disease',
    36: 'Tomato mosaic virus',
    37: 'Tomato Yellow Leaf Curl Virus',
}



# %%
def extract_hu_features(image):
    gray_img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_img)
    huMoments = cv2.HuMoments(moments)
    huMoments_new = []
    for i in range(0,7):
        moment = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        huMoments_new.append("{:.8f}".format(moment))
    return huMoments_new

# %%
def extract_haralick_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

# %%
def AssignBit(image, x, y, c):   
    bit = 0  
    try:          
        if image[x][y] >= c: 
            bit = 1         
    except: 
        pass
    return bit


# takes an image and returns an LBP value
def LocalBinaryValue(image, x, y):  
    eight_bit_binary = []
    centre = image[x][y] 
    powers = [1, 2, 4, 8, 16, 32, 64, 128] 
    decimal_val = 0
    #starting from top right,assigning bit to pixels clockwise 
    eight_bit_binary.append(AssignBit(image, x-1, y + 1,centre)) 
    eight_bit_binary.append(AssignBit(image, x, y + 1, centre)) 
    eight_bit_binary.append(AssignBit(image, x + 1, y + 1, centre)) 
    eight_bit_binary.append(AssignBit(image, x + 1, y, centre)) 
    eight_bit_binary.append(AssignBit(image, x + 1, y-1, centre)) 
    eight_bit_binary.append(AssignBit(image, x, y-1, centre)) 
    eight_bit_binary.append(AssignBit(image, x-1, y-1, centre)) 
    eight_bit_binary.append(AssignBit(image, x-1, y, centre))     
    #calculating decimal value of the 8-bit binary number
    for i in range(len(eight_bit_binary)): 
        decimal_val += eight_bit_binary[i] * powers[i] 

    return decimal_val


# returns histogram of LBP image from gray image 
def extract_LBP_features(image):
    m, n,_= image.shape 
    gray_img = (cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) * 255).astype(np.uint8)  
    lbp_img = np.zeros((m, n),np.uint8) 

    for i in range(0,m): 
        for j in range(0,n): 
            lbp_img[i, j] = LocalBinaryValue(gray_img, i, j) 
    hist_lbp = np.histogram(lbp_img, bins=20)[0]

    return hist_lbp

# %%
gamma=0.5
sigma=0.56
theta_list=[0, np.pi, np.pi/2, np.pi/4, 3*np.pi/4] # Angles
phi=0
lamda_list=[2*np.pi/1, 2*np.pi/2, 2*np.pi/3, 2*np.pi/4, 2*np.pi/5] # wavelengths
def gabor(img):    
    gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    local_energy_list=[]
    mean_ampl_list=[]
    for theta in theta_list:                
        for lamda in lamda_list:
            kernel=cv2.getGaborKernel((3,3),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)
            fimage = cv2.filter2D(gray_img.astype(np.uint8), cv2.CV_8UC3, kernel)
            mean_ampl=np.sum(abs(fimage))
            mean_ampl_list.append(mean_ampl)
            local_energy=np.sum(fimage**2)
            local_energy_list.append(local_energy)
            local_energy_list.extend(mean_ampl_list)
    return local_energy_list

# %%
def ColorHistogram(img):
    features = []        
    data = np.vstack((img[:,:,0].flat, img[:,:,1].flat, img[:,:,2].flat)).astype(np.uint8).T                                
    hist, _ = np.histogramdd(data, bins=8, range=((0,256),(0,256),(0,256)))
    hist = hist / (np.linalg.norm(hist) + 1e-16)
    hist= hist.flatten()
    features.extend(hist)
    return features

# %%
def extract_all(images):
    hu_features = []
    haralick_features = []
    LBP_features = []
    gabor_features = []
    ColorHistogram_features = []
    
    for img in images:
        hu_features.append(extract_hu_features(img))
        haralick_features.append(extract_haralick_features(img))
        LBP_features.append(extract_LBP_features(img))
        gabor_features.append(gabor(img))
        ColorHistogram_features.append(ColorHistogram(img))
        
    LBP_features = preprocessing.normalize(LBP_features)
    hu_features = preprocessing.normalize(hu_features)
    haralick_features = preprocessing.normalize(haralick_features)
    gabor_features = preprocessing.normalize(gabor_features)
    ColorHistogram_features = preprocessing.normalize(ColorHistogram_features)

    return np.hstack((LBP_features, hu_features, haralick_features, gabor_features, ColorHistogram_features))
    
