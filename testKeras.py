import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

a = np.array([[1, 2], [3, 4]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)

print (sum0)
print (sum1)

#######

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=['accuracy'])

# Generate dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(data, labels, epochs=10, batch_size=32)