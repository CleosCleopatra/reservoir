import numpy as np
import matplotlib.pyplot as plt

# --- 0. Parameters ---
np.random.seed(7)  # Ensure reproducibility

dt = 0.02
num_input_neurons = 3
num_reservoir_neurons = 500
num_output_neurons = 3
variance1 = 0.002
variance2 = 2 / num_reservoir_neurons
k_val = 0.01
washout = 100
predict_steps = 500

# --- 1. Load Data ---
train_data = np.loadtxt('training-set.csv', delimiter=',')
test_data = np.loadtxt('test-set-9.csv', delimiter=',')





# Plot training set
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(train_data[0], train_data[1], train_data[2], lw=0.8)
ax.set_title("True Lorenz Attractor (Training Set)")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()

# Plot test set
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(test_data[0], test_data[1], test_data[2], lw=0.8)
ax.set_title("True Lorenz Attractor (Test Set)")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()






train_length = train_data.shape[1]
test_length = test_data.shape[1]

# --- 2. Normalize Data for Stability ---
# Comment out normalization if your instructor requires raw data.
train_mean = np.mean(train_data, axis=1, keepdims=True)
train_std = np.std(train_data, axis=1, keepdims=True)
train_data_norm = (train_data - train_mean) / train_std
test_data_norm = (test_data - train_mean) / train_std  # Use training mean/std

# --- 3. Initialize Weights ---
input_weight = np.random.normal(0.0, np.sqrt(variance1), size=(num_reservoir_neurons, num_input_neurons))
reservoir_weight = np.random.normal(0.0, np.sqrt(variance2), size=(num_reservoir_neurons, num_reservoir_neurons))

# Scale reservoir weights to spectral radius ~0.9 (crucial for echo state property)
eigs = np.linalg.eigvals(reservoir_weight)
spectral_radius = np.max(np.abs(eigs))
desired_radius = 1.051
reservoir_weight *= (desired_radius / spectral_radius)

# --- 4. Training Pass ---
r = np.zeros((num_reservoir_neurons, train_length + 1))
for t in range(train_length):
    x_t = train_data_norm[:, t]
    r[:, t + 1] = np.tanh(reservoir_weight @ r[:, t] + input_weight @ x_t)

dropped_r = r[:, washout + 1:train_length]  # Drop washout
dropped_train = train_data_norm[:, washout + 1:train_length]
print(dropped_r.shape)
print(dropped_train.shape)

# Ridge regression for output weights
part=dropped_r @ dropped_r.T + k_val * np.identity(num_reservoir_neurons)
dropped=dropped_train @ dropped_r.T
W_out = dropped @ np.linalg.inv(part)

# --- 5. Testing Pass (Teacher-forcing on test data) ---
r_testing = np.zeros((num_reservoir_neurons, test_length + 1))
o_testing = np.zeros((num_output_neurons, test_length + 1))
for t in range(test_length):
    r_testing[:, t + 1] = np.tanh(reservoir_weight @ r_testing[:, t] + input_weight @ test_data_norm[:, t])
    o_testing[:, t + 1] = W_out @ r_testing[:, t + 1]




o_testing_denorm = o_testing * train_std + train_mean

# Assuming o_test is the output from your model during testing, shape: (3, test_length)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(o_testing_denorm[0], o_testing_denorm[1], o_testing_denorm[2], lw=0.8)
ax.set_title("Model Output During Test (Teacher-forced)")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(o_testing_denorm[0], o_testing_denorm[1], o_testing_denorm[2], lw=0.8)
ax.set_title("Model Output During Test (Teacher-forced)")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()




# --- 6. Autonomously Predict 500 Steps ---
r_pred = np.zeros((num_reservoir_neurons, predict_steps + 1))
o_pred = np.zeros((num_output_neurons, predict_steps + 1))

# Start with last test state
r_pred[:, 0] = r_testing[:, test_length]
o_old = o_testing[:, test_length]

for t in range(predict_steps):
    r_pred[:, t + 1] = np.tanh(reservoir_weight @ r_pred[:, t] + input_weight @ o_old)
    o_pred[:, t + 1] = W_out @ r_pred[:, t + 1]
    o_old = o_pred[:, t + 1]

# After prediction
#o_pred = o_pred * train_std[:, 0].reshape(-1, 1) + train_mean[:, 0].reshape(-1, 1)


# --- 7. Save y-component as prediction.csv (de-normalized) ---
# If you normalized, de-normalize y predictions for submission
#y_pred_norm = o_pred[1, 1 : predict_steps + 1]  # Shape: (500,)
#y_pred = y_pred_norm * train_std[1, 0] + train_mean[1, 0]
o_pred_denorm = o_pred * train_std + train_mean
y_pred = o_pred_denorm[1, 1 : predict_steps + 1]

np.savetxt('prediction.csv', y_pred.reshape(1, -1), delimiter=",")

# --- 8. Optional: Plot Prediction ---
plt.figure(figsize=(8, 6))
plt.plot(y_pred)
plt.title("Predicted y-component (Lorenz Reservoir)")
plt.xlabel("Time step")
plt.ylabel("y")
plt.show()


predicted_x = o_pred_denorm[0, 1:501]
predicted_y = o_pred_denorm[1, 1:501]
predicted_z = o_pred_denorm[2, 1:501]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(predicted_x, predicted_y, predicted_z, lw=0.8)

ax.set_title("Lorenz Attractor Prediction")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()




fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(test_data[0], test_data[1], test_data[2], lw=0.8, label="True Test Data")
ax.plot(predicted_x, predicted_y, predicted_z, lw=0.8, label="Model Prediction")
ax.set_title("True vs Predicted Lorenz Attractor")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.legend()
plt.show()



threshold=0.5
true_y = test_data[1, :]
pred_y = o_testing_denorm[1, 1:]
# If you normalized, de-normalize before saving
#pred_y = pred_y * train_std[1, 0] + train_mean[1, 0]

error = np.abs(true_y - pred_y)
window = np.argmax(error > threshold)  # e.g., threshold=0.5
print(f"Prediction window: {window*dt} seconds")