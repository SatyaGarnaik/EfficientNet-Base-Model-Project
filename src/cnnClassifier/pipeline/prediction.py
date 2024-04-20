import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    
    
    def predict(self):
        ## load model
        classes_dir = ["Covid-19","Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"]
        #model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (350,350))
        test_image = image.img_to_array(test_image)/255
        test_image = np.array([test_image])
        result = np.argmax(model.predict(test_image))
        print("Predicted Label:", classes_dir[result])
        return [{ "image" : classes_dir[result]}]

        