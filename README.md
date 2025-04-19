# skin_disease_app
A Streamlit-based web application that classifies skin lesions into one of seven categories using a Convolutional Neural Network (CNN) trained on the HAM10000 dataset. Users can upload a skin image, and the model predicts the most likely disease class along with the prediction confidence.
# Features
1. Upload skin lesion images (.jpg, .jpeg, .png)
2. Real-time prediction of 7 skin disease classes:
    akiec – Actinic keratoses and intraepithelial carcinoma
    bcc – Basal cell carcinoma
    bkl – Benign keratosis-like lesions
    df – Dermatofibroma
    mel – Melanoma
    nv – Melanocytic nevi
    vasc – Vascular lesions
    Visual confidence score for predictions
    User-friendly UI powered by Streamlit

# Model Info:
    Architecture: Custom CNN
    Dataset Link : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    Input size: 224x224
    Trained using TensorFlow and Keras
    Model file: skin_disease_model.h5

# How to Run :
1. Download the datset from Kaggle.
2. Run the train_model.py file. --- python train_model.py
3. The skin_disease_model.h5 file is created
4. Then run the app.py file --- python -m streamlit app.py

# Output: 
Output will show 3 things - 
1. class
2. diesease name
3. confidence score
   
