import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = {
    "study_hours":[
        1,2,2,3,3,3,4,4,4,5,
        5,5,6,6,6,7,7,7,8,8
    ],

    "attendance": [
        50,55,60,58,65,68,65,70,75,68,
        75,80,72,78,85,80,85,90,88,95
    ],

    "result": [
        0,0,0,0,0,0,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1
    ]
}

df = pd.DataFrame(data)

X = df[["study_hours","attendance"]]
y = df["result"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y, 
    test_size=0.25,
    random_state=42 
 )

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)),  

    tf.keras.layers.Dense(1, activation='sigmoid')
])  

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

model.fit(
    X_train,
    Y_train, 
    epochs=50, 
    verbose=0
    )

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

predictions = model.predict(X_test)

for actual, predicition in zip(Y_test, predictions):
    predicted_class = 1 if predicition[0]>= 0.5 else 0

    print(
        "Actual", actual, 
        "predicted", predicted_class, 
        "probability",predicition[0]
        )

print("Acuraccy:", accuracy)