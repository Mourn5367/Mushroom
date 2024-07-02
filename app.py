from flask import Flask, request, render_template
from flask_restx import Api, Resource, fields
import pickle
import numpy as np
import joblib
import sklearn
app = Flask(__name__)
api = Api(app, version='1.0', title='Model API', description='API for model predictions')
ns = api.namespace('predict', description='Prediction operations')

model = joblib.load('mushroom_model_1.pkl') 
# model.predict(X)

# Define the model input
prediction_model = api.model('Prediction', {
    'cap_color': fields.String(required=True, description='Cap color'),
    'odor': fields.String(required=True, description='Odor'),
    'habitat': fields.String(required=True, description='Habitat')
})

# Convert the feature values to numerical representation
feature_mapping = {
    'cap_color': {'b': 0, 'e': 1, 'g': 2, 'h': 3, 'k': 4, 'n': 5, 'o': 6, 'p': 7, 'r': 8, 'u': 9, 'w': 10, 'y': 11},
    'odor': {'a': 0, 'c': 1, 'f': 2, 'i': 3, 'm': 4, 'n': 5, 'p': 6, 's': 7, 'y': 8},
    'habitat': {'d': 0, 'g': 1, 'l': 2, 'm': 3, 'p': 4, 'u': 5, 'w': 6}
}

@ns.route('/')
class PredictionResource(Resource):
    @ns.doc('predict')
    @ns.expect(prediction_model)
    def post(self):
        data = request.json
        cap_color = feature_mapping['cap_color'][data['cap_color']]
        odor = feature_mapping['odor'][data['odor']]
        habitat = feature_mapping['habitat'][data['habitat']]

        # Assuming the model takes a 1D array as input
        features = np.array([[cap_color, odor, habitat]])
        
        # Make prediction using the loaded model
        prediction = model.predict(features)

        result = {
            'cap_color': data['cap_color'],
            'odor': data['odor'],
            'habitat': data['habitat'],
            'prediction': prediction[0]
        }
        return result, 200

api.add_namespace(ns, '/predict')

@app.route('/test')
def test():
    return render_template("test.html")

if __name__ == '__main__':
    app.run(debug=True)