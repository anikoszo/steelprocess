import pandas as pd
from flask import Flask, render_template, jsonify
import pickle

app = Flask(__name__)

# sample short processes to simulate how the model would predict
data = [10.12, 10.23, 10.32, 10.44, 10.55, 10.62, 10.71, 10.72, 10.73, 10.74,
        10.15, 10.30, 10.34, 10.41, 10.57, 10.68, 10.70, 10.71, 10.74, 10.8,
        10.12, 10.23, 10.32, 10.44, 10.55, 10.62, 10.71, 10.72, 10.73, 10.74,
        10.15, 10.30, 10.34, 10.41, 10.57, 10.68, 10.70, 10.71, 10.74, 10.8
        ]
gas_measurement_index = 0

with open('data/model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_gas_measurement')
def get_gas_measurement():
    global gas_measurement_index

    # Simulate random gas measurements
    if gas_measurement_index < len(data):
        gas_measurement = data[gas_measurement_index]
        gas_measurement_index += 1
    else:
        gas_measurement_index = 0
        gas_measurement = data[gas_measurement_index]
        gas_measurement_index += 1

    input_df = pd.DataFrame([{'gas1': gas_measurement}])
    # Check if the process should be continued or stopped based on the predicted value
    prediction = model.predict(input_df)

    if prediction[0] == "Need more heat":
        return jsonify({'message': 'Need more heat',
                        'color': 'orange',
                        'gas_measurement_value': gas_measurement,
                        'prediction': prediction[0]})
    elif prediction[0] == "Too much heat":
        return jsonify({'message': 'Too much heat',
                        'color': 'red',
                        'gas_measurement_value': gas_measurement,
                        'prediction': prediction[0]})
    else:
        return jsonify(
            {'message': 'Process should stop here',
             'color': 'green',
             'gas_measurement_value': gas_measurement,
             'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
