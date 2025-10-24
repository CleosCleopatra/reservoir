import numpy as np

np.random.seed(42)
#Initalise parameters and matrices 
#Parameters
num_input_neurons=3
num_reservoir_neurons=500
num_output_neurons=3
variance1=0.002
variance2=2/500
k_val=0.01
washout=100
prediction=500
scaling=0.1

#load data
train_data=np.loadtxt('training-set.csv', delimiter=',')
test_data=np.loadtxt('test-set-9.csv', delimiter=',')
train_length=train_data.shape[1]
test_length=test_data.shape[1]


#Input weights are Gaussian with variance 0.002, 
# reservoir weight Gaussian with variance 2/500
# Input weight matrix size: (N inputs* N reservoir)
input_weight=np.random.normal(0.0, np.sqrt(variance1), (num_reservoir_neurons, num_input_neurons)) #normal and gauss are the same, right?
# Reservoir weights matrix
reservoir_weight=np.random.normal(0.0, np.sqrt(variance2), (num_reservoir_neurons, num_reservoir_neurons))


#Initalise reservoir state r_0=0
r=np.zeros((num_reservoir_neurons, train_length+1))

#Training pass:
#Prepater inputs and targets
#For each t:
for t in range(train_length):
    x_t=train_data[:,t]
    r[:, t+1]=np.tanh(reservoir_weight@ r[:, t]+input_weight @ x_t)
    

#Drop the first 100 steps
dropped_r=r[:, washout+1:train_length]
dropped_train=train_data[:, washout+1: train_length] 


W_out=(dropped_train@dropped_r.T) @np.linalg.inv(dropped_r @ dropped_r.T + k_val * np.eye(num_reservoir_neurons))

r_testing=np.zeros((num_reservoir_neurons, test_length+1))
o_testing=np.zeros((num_output_neurons, test_length+1))

for t in range(test_length):
    r_testing[:, t+1]=np.tanh(reservoir_weight@r_testing[:, t]+ input_weight@test_data[:, t])
    o_testing[:, t+1]=W_out @ r_testing[:, t+1]

#predict 500 time steps

r_pred=np.zeros((num_reservoir_neurons, 501))
o_pred=np.zeros((num_output_neurons, 501))
r_pred[:,0]=r_testing[:, test_length] 


o_old=o_testing[:, test_length]
#For each t:
for t in range(500):
    r_pred[:, t+1]=np.tanh(reservoir_weight@ r_pred[:, t]+input_weight @ o_old)
    o_pred[:, t+1]=W_out @ r_pred[:, t+1]
    o_old=o_pred[:, t+1]

predicted_values=o_pred[1, 1:501]
np.savetxt('prediction.csv', predicted_values.reshape(1,-1), delimiter=",")

import matplotlib.pyplot as plt

predicted_x = o_pred[0, 1:501]
predicted_y = o_pred[1, 1:501]
predicted_z = o_pred[2, 1:501]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(predicted_x, predicted_y, predicted_z, lw=0.8)

ax.set_title("Lorenz Attractor Prediction")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()