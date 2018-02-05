from keras.datasets import imdb
from keras import models
from keras import layers

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

print (x_train.shape)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(25000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_test, y_test))