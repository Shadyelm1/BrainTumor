from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from keras.models import load_model

app = Flask("BrainScanAI", template_folder='templates')

model = load_model('BrainTumor.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.abspath(os.path.dirname(__file__))

        image_path = os.path.join(basepath, 'uploads', secure_filename(f.filename)).replace('\\', '\\\\')
        f.save(image_path)

        image = cv2.imread(image_path)
        img = cv2.resize(image, (64, 64))
        img = img.astype('float32') / 255
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        print("Prediction probabilities:", prediction)

        # Extract the class based on the highest probability
        predicted_class = "Yes Brain Tumor" if prediction[0][0] > 0.5 else "No Brain Tumor"
        print("Predicted class:", predicted_class)

        return predicted_class
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
