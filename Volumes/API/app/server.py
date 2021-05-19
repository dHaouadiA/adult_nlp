from flask import Flask, request, jsonify
from joblib import load
import pickle5 as pickle


#clf = load("app/clf.joblib")
#this file contain source endd point of the web server
app = Flask(__name__)


@app.route("/predict", methods=["POST"])


def predict():
    """Takes a POST request with a key of \"text\" and the text to be classified."""
    data_dict = request.get_json()
    import keras
    import model_ap as mp
        # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    reconstructed_model = keras.models.load_model("model_Bi_LSTM.h5")
    url = data_dict["textUrl"]
    print(url)
    return jsonify({"result":str(mp.predict_url(url,reconstructed_model,tokenizer))
})
 
if __name__ == "__main__":
    app.run(host='0.0.0.0')