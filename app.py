from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, redirect, url_for, request
from flask import Flask,render_template,request,redirect,session


app = Flask(__name__)



# Loading Models
covid_model = load_model('C:\\Users\\basir\\OneDrive\\Desktop\\Multi-disease detection\\models\\covid_model.h5')
braintumor_model = load_model('C:\\Users\\basir\\OneDrive\\Desktop\\Multi-disease detection\\models\\Brain_Tumor.h5')
alzheimer_model = load_model('C:\\Users\\basir\\OneDrive\\Desktop\\Multi-disease detection\\models\\Alzheimer_model.h5')
pneumonia_model = load_model('C:\\Users\\basir\\OneDrive\\Desktop\\Multi-disease detection\\models\\Pneumonia_model.h5')

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

############################################# BRAIN TUMOR FUNCTIONS ################################################



def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        # Resize image to (299, 299)
        new_img = cv2.resize(new_img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        set_new.append(new_img)

    return np.array(set_new)


########################### Routing Functions ########################################s


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/covid')
def covid():
    return render_template('covid.html')



@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')



@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')



########################### Result Functions ########################################    


@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = covid_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = image.load_img('static/uploads/'+filename, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            pred = braintumor_model.predict(img)
            # Assuming pred contains probabilities for each class
            # You can use np.argmax(pred) to get the index of the class with the highest probability
            # Then, you can compare this index with the class you're interested in
            class_index = np.argmax(pred)
            if class_index == 1:  # Assuming 1 is the index for the positive class
                result = "Positive"
            else:
                result = "Negative"
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=result, gender=gender)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)






@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)
        
        

@app.route('/resulta', methods=[ 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)

            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=0, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')




# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
