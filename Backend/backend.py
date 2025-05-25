from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


all_datas = []
# Load the model
try:
    model = joblib.load("crop_recommendation_model_v1.pkl")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print("❌ Error loading model:", str(e))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        global all_datas
        global latest_prediction
        data = request.json

        features_array = [
            float(data.get("nitrogen", 0)),
            float(data.get("phosphorus", 0)),
            float(data.get("potassium", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0)),
            float(data.get("pH_Level", 0)),
            float(data.get("rainfall", 0))
        ]
       
        features = np.array(features_array).reshape(1, -1)  # Convert to 2D array for model input
        print(features_array)
        # Predict using the model
        prediction = model.predict(features).tolist()

        print("Model Prediction:", prediction)
        latest_prediction = prediction
        all_datas.append(prediction)
        all_datas.append(features_array)
        print(all_datas)

       # return jsonify({"prediction": prediction[0]})  # Send back result
        return jsonify({"message": "Prediction stored", "prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/userQuery", methods=["POST"])
def userQuery():
    try:
        
        data = request.json
        
        query = data.get("text",0)
       # store the user query 

        all_datas.append(query)

        print(all_datas)

        n, p, k, temperature, humidity, ph, rainfall = all_datas[1]

        # Formatting the inputs with labels
        formatted_inputs = f"N: {n}, P: {p}, K: {k}, Temperature: {temperature}, Humidity: {humidity}, PH: {ph}, Rainfall: {rainfall}"

        # Creating the final prompt string
        prompt = f"Retrieve precise agricultural recommendations from the vector database based on the user inputs: {formatted_inputs}. The response should be strictly aligned with the predicted model output: {all_datas[0]} and should also address the user's additional query: {all_datas[2]}. The answer must be clear, well-structured, and formatted for readability. Provide insights on the best irrigation practices suitable for the crop, considering soil type, climate, and water requirements. Include secondary crop recommendations that improve soil health and productivity. Specify the appropriate fertilizers, their dosage, and application frequency for optimal growth. Offer detailed care and maintenance tips, including pest control methods, pruning techniques, and disease prevention strategies. Additionally, provide any other relevant agronomic insights that enhance crop yield and sustainability. Ensure that all information is sourced strictly from the vector database and avoid generating answers beyond retrieved data."

      
        all_datas.append(prompt)
        print(prompt)
        
        return jsonify({ "prompt": prompt})
    except Exception as e:
        return jsonify({"error": str(e)})
    
      


@app.route("/get-latest-prediction", methods=["GET"])
def get_latest_prediction():
    if all_datas is not None:
        return jsonify({"latest_prediction": all_datas})
    else:
        return jsonify({"error": "No prediction available"}), 404
if __name__ == "__main__":
    app.run(debug=True,port=5001)