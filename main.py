import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

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

data_train, data_test, result_train, result_test = train_test_split(data, result, test_size=0.2, random_state=42)


model = keras.Sequential([
    keras.layers.Dense(5, activation='relu', input_shape=(4,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

model.fit(data_train, result_train, epochs=100, batch_size=32, validation_split=0.2)

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

test_loss, test_accuracy = model.evaluate(data_test, result_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
print(f'Test loss: {test_loss:.2f}%')