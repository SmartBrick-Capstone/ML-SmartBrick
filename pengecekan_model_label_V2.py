from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
# sesuaikan dengan Yang sudah diupload
model = load_model("keras_Model.h5", compile=False)

# Load the labels
# Sesuikan dengan yang sudah di upload berdasarkan data terbaru
class_names = open("labels.txt", "r").readlines()


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("yo.jpg").convert("RGB")

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


# Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)
# text="Plastik"
# x = class_name[0].find(text)
# if confidence_score>0.9 and x:
#     print("layak")
# else :
#     print("tidak layak")
    # if class_name[0]:
    #     print("Ini layak dijadikan ecobrick")
    # else:
    #     print("Ini tidak layak dijadikan ecobrick")

class_index = np.argmax(prediction)
predicted_class = class_names[class_index]  # Ganti 'class_names' dengan daftar nama kelas Anda
confidence = prediction[0][class_index] * 100  # Konversi ke presentase kepercayaan

result = ''
if(confidence_score>0.9):
    if(predicted_class == class_names[5] or class_names[1]):
        result = 'Benda ini layak dijadikan eco-brick '+ predicted_class
    else:
        result = ' Benda ini tidak layak dijadikan eco-brick ' + predicted_class
else:
    result = 'Object tidak diketahui'

print(result)