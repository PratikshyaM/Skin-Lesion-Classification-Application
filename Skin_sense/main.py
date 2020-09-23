from flask import Flask, request, render_template#, Response
import numpy as np
import os
import string
import random

#PATH = "C:/Users/Dell/Desktop/HackerEarth Healthcare/Project/others/"
PATH = "others/"
OUTPUT_DIR = 'static'

app = Flask(__name__)

def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'

def get_prediction(image_path):
    import tensorflow as tf
    from model import SkinLesionTypeDetectionModel
    model = SkinLesionTypeDetectionModel(PATH+"model.json", PATH+"model.h5")
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    pred1, pred2, pred3 = model.predict_skin_lesion_type(image)

    print(pred1, pred2, pred3)
    return pred1, pred2, pred3

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                image_path = os.path.join(OUTPUT_DIR, generate_filename())
                uploaded_file.save(image_path)
                
                """""" #Remove the following three lines
                pred1 = ('Melanoma', 0.5374107) 
                pred2 = ('Basal Cell Carcinoma', 0.21953377) 
                pred3 = ('Dermatofibroma', 0.1933243)
                """"""
                #Uncomment this
                #pred1, pred2, pred3 = get_prediction(image_path)
                result = {
                    'highest_class_name': pred1[0],
                    'highest_prob':np.round(pred1[1]*100),
                    
                    'second_highest_class_name': pred2[0],
                    'second_highest_prob':np.round(pred2[1]*100),
                    
                    'third_highest_class_name': pred3[0],
                    'third_highest_prob':np.round(pred3[1]*100),
                    'path_to_image': image_path
                }
                
                return render_template('show.html', result=result)
    return render_template('index.html')

'''@app.route('/skin_lesion_image/<img>',methods=['POST'])
def skin_lesion_image(img):
    print("here")
    return Response(gen(img))'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,use_reloader=False)
