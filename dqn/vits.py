from transformers import ViTFeatureExtractor, ViTModel
from transformers import AutoFeatureExtractor, ViTMAEModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image


# ViT base 8
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')
inputs = feature_extractor(images=image, return_tensors="pt")
inputs['pixel_values'].shape
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

last_hidden_states.flatten().shape


# ViT base 16
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

last_hidden_states.shape
last_hidden_states.flatten().shape



# MAE
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

last_hidden_states.shape
last_hidden_states.flatten().shape




# Test
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=True)

img = image.load_img("/Users/czarrar/Downloads/000000039769.jpg", target_size=(224, 224))
x = image.img_to_array(img) # This is a PIL Image
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
features.shape

model.summary()
