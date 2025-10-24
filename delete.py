
import numpy as np
from scipy.linalg import eigvals
import matplotlib.pyplot as plt


np.random.seed(42)

train_data = np.loadtxt('training-set.csv', delimiter=',')
test_data = np.loadtxt('test-set-9.csv', delimiter=',')
T_train=train_data.shape[1]
T_test=test_data.shape[1]


# Reservoir Computing Parameters
num_input=3
num_reservoir = 500  # Number of reservoir neurons
variance_in=0.002
variance_res=2/num_reservoir
ridge_param=0.01
washout=100
predict_steps=500
input_scaling = 0.1   

# Initialize reservoir
W_in = np.random.normal(0, np.sqrt(variance_in), (num_reservoir, num_input))
W_res = np.random.normal(0, np.sqrt(variance_res), (num_reservoir, num_reservoir)) - 0.5
spectral_radius=1.05
eigs=eigvals(W_res)
current_radius=np.max(np.abs(eigs))
W_res = spectral_radius * W_res / max(abs(eigvals(W_res)))  # Adjust spectral radius

r=np.zeros((num_reservoir, T_train+1))
for t in range(T_train):
    r[:, t+1]=np.tanh(W_res@r[:, t]+ W_in @ train_data[:, t])

# Train the reservoir
states = r[:, washout+1:T_train]  # To store reservoir states
targets=train_data[:, washout+1: T_train]

W_out=targets@states.T @ np.linalg.inv(states @states.T + ridge_param*np.eye(num_reservoir))

r_test=np.zeros((num_reservoir, T_test+1))
o_test=np.zeros((3, T_test+1))

for t in range(T_test):
    #input_signal = x_train[t]
    r_test[:, t+1]=np.tanh(W_res@r_test[:, t]+W_in@test_data[:, t])
    o_test[:, t+1]=W_out @ r_test[:, t+1]


r_pred=np.zeros((num_reservoir, predict_steps+1))
o_pred=np.zeros((3, predict_steps+1))
r_pred[:, 0]=r_test[:, T_test]
o_pred[:, 0]=o_test[:, T_test]
for t in range(predict_steps):
    r_pred[:, t+1]=np.tanh(W_res@r_pred[:, t]+W_in @ o_pred[:, t])
    o_pred[:, t+1]=W_out@r_pred[:, t+1]

x_pred=o_pred[0, 1:predict_steps+1]
y_pred=o_pred[1, 1:predict_steps+1]
z_pred=o_pred[2, 1:predict_steps+1]
np.savetxt('prediction.csv', y_pred.reshape(1, -1), delimiter=",")

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_pred, y_pred, z_pred, lw=0.8)

ax.set_title("Lorenz Attractor Prediction")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()





