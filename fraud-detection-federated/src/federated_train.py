
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# Load dataset
df = pd.read_csv("../data/synthetic_data.csv")
features = ["amount", "is_international", "merchant_id"]
target = "is_fraud"

X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.int32)

# Simulate 3 clients evenly
def split_clients(X, y, num_clients=3):
    split_X = np.array_split(X, num_clients)
    split_y = np.array_split(y, num_clients)
    return list(zip(split_X, split_y))

client_data = split_clients(X, y, num_clients=3)

def create_tf_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((X, y)).shuffle(buffer_size=10).batch(4)

federated_train_data = [create_tf_dataset(x, y) for x, y in client_data]

# === Model Builder ===
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(len(features),)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# === Federated Process ===
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.05),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam()
)

state = iterative_process.initialize()

# Train for several rounds
NUM_ROUNDS = 10
for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f"Round {round_num}: {metrics}")

# Evaluate global model
def evaluate_model(state, dataset):
    keras_model = create_keras_model()
    model_weights = state.model.trainable
    tff.learning.models.assign_weights_to_keras_model(keras_model, model_weights)
    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    result = keras_model.evaluate(dataset.batch(8), verbose=0)
    print(f"\nGlobal Model Evaluation: Loss={result[0]:.4f}, Accuracy={result[1]:.4f}")

# Combine all client data for evaluation
full_X = np.vstack([x for x, _ in client_data])
full_y = np.hstack([y for _, y in client_data])
full_dataset = tf.data.Dataset.from_tensor_slices((full_X, full_y))

evaluate_model(state, full_dataset)
