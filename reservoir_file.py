import numpy as np

np.random.seed(5)
#Initalise parameters and matrices 
#Parameters
dt=0.02
num_input_neurons=3
num_reservoir_neurons=500
num_output_neurons=3 #????
variance1=0.002
variance2=2/500
k_val=0.01
washout=100

#load data
train_data=np.loadtxt('training-set.csv', delimiter=',')
test_data=np.loadtxt('test-set-9.csv', delimiter=',')
train_length=train_data.shape[1]
test_length=test_data.shape[1]

print(train_data.shape)
print(train_data[1].shape)



mean = np.mean(train_data, axis=1, keepdims=True)
std = np.std(train_data, axis=1, keepdims=True)
train_data_n = (train_data - mean) / std
test_data_n = (test_data - mean) / std









#Input weights are Gaussian with variance 0.002, 
# reservoir weight Gaussian with variance 2/500
# Input weight matrix size: (N inputs* N reservoir)
input_weight=np.random.normal(0.0, np.sqrt(variance1), size=(num_reservoir_neurons, num_input_neurons)) #normal and gauss are the same, right?
# Reservoir weights matrix
reservoir_weight=np.random.normal(0.0, np.sqrt(variance2), size=(num_reservoir_neurons, num_reservoir_neurons))



desired_radius=0.9
eigs = np.linalg.eigvals(reservoir_weight)
spectral_radius = np.max(np.abs(eigs))
reservoir_weight *= (desired_radius / spectral_radius)  # desired_radius ≈ 0.9




#Training pass:
#Prepater inputs and targets
import math


#Initalise reservoir state r_0=0
r=np.zeros((num_reservoir_neurons, train_length+1))
r_without_first=np.zeros((num_reservoir_neurons, train_length))

#For each t:
for t in range(train_length):
    x_t=train_data[:,t]
    r[:, t+1]=np.tanh(reservoir_weight@ r[:, t]+input_weight @ x_t)
    r_without_first[:,t]=r[:, t+1]
    

#Drop the first 100 steps
dropped_r=r_without_first[:, washout:]
dropped_train=train_data[:, washout:].copy() #should i keep the copy thing



#train_data=np.array(train_data)
#r=np.array(r)
#print(f"r_shape is {r.shape}")

W_out=(dropped_train@dropped_r.T) @np.linalg.inv(dropped_r @ dropped_r.T + k_val * np.identity(num_reservoir_neurons))

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