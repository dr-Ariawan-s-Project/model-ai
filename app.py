from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/bpnn_data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking, optional but recommended
db = SQLAlchemy(app)

model = keras.Sequential([
    keras.layers.Dense(18, activation='relu', input_shape=(9,)),
    keras.layers.Dense(18, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

class Data(db.Model):
    id: Mapped[int]  = mapped_column(Integer, primary_key=True)
    season: Mapped[str] = mapped_column(String)
    age: Mapped[float] = mapped_column(Float)
    childishdiseases: Mapped[str] = mapped_column(String)
    accidentorserioustrauma: Mapped[str] = mapped_column(String)
    surgicalintervention: Mapped[str] = mapped_column(String)
    highfeversinthelastyear: Mapped[str] = mapped_column(String)
    frequencyofalcoholconsumption: Mapped[str] = mapped_column(String)
    smokinghabit: Mapped[str] = mapped_column(String)
    numberofhoursspentsittingperday: Mapped[float] = mapped_column(Float)
    diagnosis: Mapped[str] = mapped_column(String)
    season_cd: Mapped[float] = mapped_column(Float)
    chld_cd: Mapped[float] = mapped_column(Float)
    accd_trauma_cd: Mapped[float] = mapped_column(Float)
    surgical_cd: Mapped[float] = mapped_column(Float)
    fever: Mapped[float] = mapped_column(Float)
    alcohol_cd: Mapped[float] = mapped_column(Float)
    smoking_cd: Mapped[float] = mapped_column(Float)
    diagnosis_cd: Mapped[float] = mapped_column(Float)

@app.route('/learn', methods=['POST'])
def learn():
    res =  db.session.execute(db.select(Data)).scalars().all()
    data_db = []
    data_result = []
    for item in res:
        item_push = [
            item.age,
            item.season_cd,
            item.chld_cd,
            item.accd_trauma_cd,
            item.surgical_cd,
            item.fever,
            item.alcohol_cd,
            item.smoking_cd,
            item.numberofhoursspentsittingperday
        ]
        data_db.append(item_push)
        
        data_result.append(item.diagnosis_cd)
        
    data = np.array(data_db)
    result = np.array(data_result)

    data_train, data_temp, result_train, result_temp = train_test_split(data, result, test_size=0.3, random_state=42)
    data_val, data_test, result_val, result_test = train_test_split(data_temp, result_temp, test_size=0.7, random_state=42)
    
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    model.fit(data_train, result_train, epochs=150, batch_size=256, validation_data=(data_val, result_val))

    model.summary()
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            weights, biases = layer.get_weights()
            print(f'Layer Name: {layer.name}')
            print(f'Weights Shape: {weights.shape}')
            print(f'Biases Shape: {biases.shape}')
            print(f'Weights:\n{weights}')
            print(f'Biases:\n{biases}\n')
    
    test_loss, test_accuracy = model.evaluate(data_test, result_test)

    return jsonify({
        "total_data": len(data_db),
        "accuracy": test_accuracy,
        "test_loss": test_loss
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
   
    new_data = [
            req['age'],
            req['season'],
            req['childishds'],
            req['trauma'],
            req['surgical'],
            req['highfever'],
            req['alcohol'],
            req['smoking'],
            req['sitting']
        ]
    
    new_data = np.array([new_data])
    predicted_result = model.predict(new_data)

    class_label = 0.0
    probability = 0.0
    
    
    for i, prediction in enumerate(predicted_result):
        class_label = 1 if prediction >= 0.5 else 0  # Assuming binary classification
        probability = prediction[0]
    
    return jsonify({
        "class": class_label,
        "probability": float(probability)
    }), 200

if __name__ == '__main__':
    app.run()