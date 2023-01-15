import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import requests
import tensorflow as tf

# Import Flask API
from flask import Flask, request
server = Flask(__name__)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')


def read_tensor_from_image_url(url,
                               input_height=224,
                               input_width=224,
                               input_mean=0,
                               input_std=255):
    image_reader = tf.image.decode_jpeg(
        requests.get(url).content, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize(dims_expander,[input_height,input_width], method='bilinear',antialias=True, name = None)
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    return normalized

@server.route("/prediction", methods=['POST'])
def keras():
    #Get all the values in your POST request. 
    apikey = request.args.get('apikey')
    image = request.args.get('url')
    #Check for API key access  --> Very makeshift manual solution. Totally not fit for production levels. 
    #Change this if you're using this method.
    if apikey == '123-456-7890-0987-654321': 
         #Follow all the neccessary steps to get the prediction of your image. 
        image = read_tensor_from_image_url(image)
        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        #Return the prediction and a 200 status
        return prediction, 200

    else:
        #If the apikey is not the same, then return a 400 status indicating an error.
        return "Not valid apikey", 400 
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.

# # Replace this with the path to your image
# image = Image.open('test_photo.jpg')

# #resize the image to a 224x224 with the same strategy as in TM2:
# #resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.ANTIALIAS)



# # display the resized image
# image.show()