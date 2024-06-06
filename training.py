import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as k
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import Sequential

df = pd.read_csv('Data/features_3_sec.csv')
df = df.drop(labels='filename', axis=1)

#preprocessing of data
class_list = df.iloc[:, -1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)

# print(class_list)
# print(y)

#scaling the features
fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype = float))

# #dividing data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(len(y_train))
print(len(y_test))

# #TRAINING MODEL, CNN algorithm (convolutional neural networks)
def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=[k.metrics.SparseCategoricalAccuracy()])
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

def plotValidate(history):
    print("Validation Accuracy", max(history.history["val_sparse_categorical_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(9,5))
    plt.show()

model = k.Sequential([
    k.layers.Dense(512, activation='relu', input_shape=(X_train.shape[128,128])),
    k.layers.Dropout(0.2),

    k.layers.Dense(256, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(64, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(10, activation='softmax'),
])

print(model.summary())
model_history = trainModel(model=model, epochs=200, optimizer='adam')
plotValidate(model_history)

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print("The test loss is: ", test_loss)
print("The best test accuracy is: ", test_acc*100)