from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import numpy as np



def prediction(imgs,model):
    model = load_model(model)
    img = image.load_img(imgs, target_size=(48, 48)) #resize image to 48x48
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Make predictions
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    emotion_classes = ['angry',  'happy', 'neutral', 'sad']
    return emotion_classes[predicted_class]