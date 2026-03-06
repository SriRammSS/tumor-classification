import os
import uuid
import flask
import urllib
from PIL import Image
import tf_keras
from flask import Flask , render_template  , request , send_file
from tf_keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf_keras.models.load_model(os.path.join(BASE_DIR , 'model.hdf5'))


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['Adenocarcinoma' ,'Large cell carcinoma', 'Normal' , 'Squamous cell carcinoma' , 'G' ,'M' ,'N', 'S' ,'B' ,'H']


def predict(filename , model):
    img = load_img(filename , target_size = (224 , 224))
    img = img_to_array(img)
    img = img.reshape(1 , 224 ,224 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(4):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result




@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }
                if class_result[0] == 'Adenocarcinoma':
                  recommendation_text = "The prognosis for adenocarcinoma of the lung depends on the stage of the cancer and the patient's overall health. However, the overall five-year survival rate for all stages of adenocarcinoma of the lung is approximately 30-35%.Treatment options include surgery, chemotherapy, radiation therapy, targeted therapy, and immunotherapy."
                elif class_result[0] == 'Large cell carcinoma':
                  recommendation_text = "The prognosis for large cell carcinoma of the lung is generally poor. The 5-year survival rate for patients with large cell carcinoma is approximately 16%. Treatment options depend on the size and location of the tumor, and may include surgery, chemotherapy, and radiation therapy."
                elif class_result[0] == 'Normal':
                  recommendation_text = "Congrats!! Your reports seems normal, but it is advisable to consult your Pulmonologist and Oncologist"
                elif class_result[0] == 'Squamous cell carcinoma':
                  recommendation_text = "The five year survival rate for localized squamous cell carcinoma of the lung (cancer that has not spread to other parts of the body) is approximately 50-60%. The five year survival rate for regional squamous cell carcinoma of the lung (cancer that has spread to nearby lymph nodes and/or organs) is approximately 30-40%. The five year survival rate for distant squamous cell carcinoma of the lung (cancer that has spread to organs away from the lungs) is approximately 5%.Treatment for squamous cell carcinoma of the lung typically involves surgical removal of the tumor and surrounding tissue, as well as radiation and/or chemotherapy to prevent or slow the growth or spread of the cancer. "
                else:
                  recommendation_text = "Unknown Aetiology"
            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions, recommendation_text=recommendation_text)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True, port=5001)


