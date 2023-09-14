import numpy as np
from tensorflow import keras

data = np.array([
    [1,1,1,1],
    [1,1,0,0],
    [0,1,1,0],
    [1,0,0,0],
    [0,0,0,0]
])
result = np.array([
    1,
    1,
    0,
    0,
    0
])

model = keras.Sequential([
    keras.layers.Dense(5, activation='relu', input_shape=(4,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

model.fit(data, result, epochs=100, batch_size=32, validation_split=0.2)

new_data = np.array([[1,0,1,0]])
predicted_result = model.predict(new_data)
print(f'Predicted result: {predicted_result[0][0]:.0f}')

model.summary()
for layer in model.layers:
    if hasattr(layer, 'weights'):
        weights, biases = layer.get_weights()
        print(f'Layer Name: {layer.name}')
        print(f'Weights Shape: {weights.shape}')
        print(f'Biases Shape: {biases.shape}')
        print(f'Weights:\n{weights}')
        print(f'Biases:\n{biases}\n')

accuracy = model.evaluate(data, result)
print(accuracy)
# print(f'Accuracy: {accuracy * 100:.2f}%')