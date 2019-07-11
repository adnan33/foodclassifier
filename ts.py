from PIL import Image
#from labels import LABEL_MAP
import numpy as np
import tempfile

from tfserve import TFServeApp
Label_map={0: 'apple_pie', 1: 'baby_back_ribs', 2: 'baklava', 3: 'beef_carpaccio', 4: 'beef_tartare', 5: 'beet_salad', 6: 'beignets', 7: 'bibimbap', 8: 'bread_pudding', 9: 'breakfast_burrito', 10: 'bruschetta', 11: 'caesar_salad', 12: 'cannoli', 13: 'caprese_salad', 14: 'carrot_cake', 15: 'ceviche', 16: 'cheesecake', 17: 'cheese_plate', 18: 'chicken_curry', 19: 'chicken_quesadilla', 20: 'chicken_wings', 21: 'chocolate_cake', 22: 'chocolate_mousse', 23: 'churros', 24: 'clam_chowder', 25: 'club_sandwich', 26: 'crab_cakes', 27: 'creme_brulee', 28: 'croque_madame', 29: 'cup_cakes', 30: 'deviled_eggs', 31: 'donuts', 32: 'dumplings', 33: 'edamame', 34: 'eggs_benedict', 35: 'escargots', 36: 'falafel', 37: 'filet_mignon', 38: 'fish_and_chips', 39: 'foie_gras', 40: 'french_fries', 41: 'french_onion_soup', 42: 'french_toast', 43: 'fried_calamari', 44: 'fried_rice', 45: 'frozen_yogurt', 46: 'garlic_bread', 47: 'gnocchi', 48: 'greek_salad', 49: 'grilled_cheese_sandwich', 50: 'grilled_salmon', 51: 'guacamole', 52: 'gyoza', 53: 'hamburger', 54: 'hot_and_sour_soup', 55: 'hot_dog', 56: 'huevos_rancheros', 57: 'hummus', 58: 'ice_cream', 59: 'lasagna', 60: 'lobster_bisque', 61: 'lobster_roll_sandwich', 62: 'macaroni_and_cheese', 63: 'macarons', 64: 'miso_soup', 65: 'mussels', 66: 'nachos', 67: 'omelette', 68: 'onion_rings', 69: 'oysters', 70: 'pad_thai', 71: 'paella', 72: 'pancakes', 73: 'panna_cotta', 74: 'peking_duck', 75: 'pho', 76: 'pizza', 77: 'pork_chop', 78: 'poutine', 79: 'prime_rib', 80: 'pulled_pork_sandwich', 81: 'ramen', 82: 'ravioli', 83: 'red_velvet_cake', 84: 'risotto', 85: 'samosa', 86: 'sashimi', 87: 'scallops', 88: 'seaweed_salad', 89: 'shrimp_and_grits', 90: 'spaghetti_bolognese', 91: 'spaghetti_carbonara', 92: 'spring_rolls', 93: 'steak', 94: 'strawberry_shortcake', 95: 'sushi', 96: 'tacos', 97: 'takoyaki', 98: 'tiramisu', 99: 'tuna_tartare', 100: 'waffles'}


# 1. Model: trained mobilenet on ImageNet that can be downloaded from
#           https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
MODEL_PATH = r"F:\Food_model.pb"


# 2. Input tensor names:
INPUT_TENSORS = ["import/input_1_1:0"]

# 3. Output tensor names:
OUTPUT_TENSORS = ["import/Softmax_1:0"]


# 4. encode function: Receives raw jpg image as request_data. Returns dict
#                     mappint import/input:0 to numpy value.
#                     Model expects 224x224 normalized RGB image.
#                     That is, [224, 224, 3]-size numpy array.
def encode(request_data):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
        f.write(request_data)
        img = Image.open(f.name).resize((224, 224))
        img = np.asarray(img) / 255.

    return {INPUT_TENSORS[0]: img}


# 5. decode function: Receives `dict` mapping import/MobilenetV2/Predictions/Softmax:0 to
#                     numpy value and builds dict with for json response.
def decode(outputs):
    p = outputs[OUTPUT_TENSORS[0]]
    # p will be a 1001 numpy array with all class probabilities.
    index = np.argmax(p)
    
    return {"class": Label_map[index-1], "prob": float(p[index])}


# Run the server
print('go')
app = TFServeApp(MODEL_PATH, INPUT_TENSORS, OUTPUT_TENSORS, encode, decode)
print('go2')
app.run('0.0.0.0', 5000)
'''for word, count in wordcount(r"F:\UPMC_Food101.tar\texts_txt\apple_pie\apple_pie_6.txt", triggers).items():
    print("Count of {}: {}".format(word, count))'''
