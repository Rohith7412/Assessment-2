import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = {
    "age": [
        22, 25, 28, 30, 32, 35, 38, 40, 42, 45,
        24, 27, 29, 31, 34, 36, 39, 41, 44, 47
    ],

    "monthly_income": [
        22000, 25000, 28000, 32000, 35000,
        40000, 45000, 50000, 55000, 60000,
        24000, 27000, 30000, 34000, 38000,
        42000, 48000, 52000, 58000, 65000
    ],

    "years_at_company": [
        1, 1, 2, 2, 3, 4, 5, 7, 8, 10,
        1, 2, 2, 3, 4, 5, 6, 7, 9, 12
    ],

    "job_satisfaction": [
        1, 1, 2, 2, 2, 3, 3, 4, 4, 4,
        1, 2, 2, 3, 3, 4, 4, 4, 4, 5
    ],

    "overtime": [
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0
    ],

    "attrition": [
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

x = df[[
    "age",
    "monthly_income",
    "years_at_company",
    "job_satisfaction",
    "overtime"
]]

y = df["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(5,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=100,
    verbose=0
)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

predictions = model.predict(X_test)

for actual, prediction in zip(y_test, predictions):
    predicted_class = 1 if prediction[0] >= 0.5 else 0

    print(
        "Actual:",
        actual,
        "Predicted:",
        predicted_class,
        "Probability:",
        prediction[0]
    )

print("Accuracy:", accuracy)