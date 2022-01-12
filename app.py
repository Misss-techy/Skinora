from flask import Flask, render_template, request, jsonify, url_for
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import urllib.parse, urllib.error, base64, os, random
from flask import redirect

# PREDICTION_KEY = os.environ['Prediction_Key']
# ENDPOINT = os.environ['Prediction_Endpoint']
# PROJECT_ID = os.environ['Project_Id']
# PUBLISH_ITERATION_NAME = os.environ['Iteration_Name']

PREDICTION_KEY = '645252b1d8ae4a039943db3103400fad'
ENDPOINT = 'https://southeastasia.api.cognitive.microsoft.com/'
PROJECT_ID = 'cfde5311-10e2-45f0-b198-0b30585c1b42'
PUBLISH_ITERATION_NAME = 'Iteration1'
LABELS = ['Acne', 'ActinicKeratosis', 'AtopicDermatitis', 'BullousDisease', 'CellulitisImpetigo','Eczema','Exanthems',
'Alopecia','Herpes','LightDiseases','Lupus','Melanoma','NailFungus','ContactDermatitis','Psoriasis','Scabies',
'SeborrheicKeratoses','SystemicDisease','TineaRingworm','UrticariaHives','VascularTumors','Vasculitis','Warts']

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diseases', methods=['GET', 'POST'])
def diseases():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('diseases.html')

@app.route('/classify', methods=['POST'])
def classify():
    body = request.get_json()
    # print(body['image_base64'])
    image_bytes = base64.b64decode(body['image_base64'].split(',')[1])
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})

    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    results = predictor.classify_image(
        PROJECT_ID, PUBLISH_ITERATION_NAME, image_bytes)
    
    predictions = {prediction.tag_name: prediction.probability for prediction in results.predictions}

    predicted = max(predictions.keys(), key=(lambda k: predictions[k]))

    return jsonify({'predicted': predicted,
                    'probability': predictions[predicted],
                    'opponent': LABELS[random.randint(0,4)]})
