from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
# sesuaikan dengan Yang sudah diupload
model = load_model("keras_Model1.h5", compile=False)

# Load the labels
# Sesuikan dengan yang sudah di upload berdasarkan data terbaru
class_names = open("labels1.txt", "r").readlines()


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("daia.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

#ini yang ditampilkan ke layar
if (confidence_score>0.9 and class_name[0]):
    print("Benda ini layak untuk dijadikan ecobrick")
else :
    print("Benda ini tidak layak dijadikan ecobrick")

# Print prediction and confidence score
#print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)
