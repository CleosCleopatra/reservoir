#resevoir
#Sample weigthts from a normal distribution with mean zero and variance N σ^2_w= 1.
#Input weights w_i^{in} were samples from normal distribution with mean zero and variance σ_{in}^2=1
#Resservoir state initalised to zero r_i(0)=0
#Reseoir dynamics was iterated through using second component x_2(t) of Ikeda time series as input x(t)
#First 100 iterations were discarded
#Energy function calculated from the next T:
#H=(1/2) \sum_{t=0}^{T-1} [y(t)-O(t)]^2 =(1/2) ||y-R^T w^{(out)]}||^w
#for time series prediction, y(t)=x(t)
#R=[r(0), ...., r(T-1)] and y=[x(0),..., x[T-1]]^T
#Minimise H to determine w^{out}:
#   With ridge regression with ridge parameter 0.001


#Resevoir computer is used to predict time series
#Output were fed into inputs
#ie c(t) in 9.35a was replaced by O(t)
#Iterating through reservoir dynamics once, one gets r(T) from r(T-1) 
#and O(T)=w^{out} \cdot r^{T}

import random
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
#Input weights are Gaussian with variance 0.002, 
# reservoir weight Gaussian with variance 2/500
# Input weight matrix size: (N inputs* N reservoir)
input_weight=np.random.normal(0.0, np.sqrt(variance1), size=(num_reservoir_neurons, num_input_neurons)) #normal and gauss are the same, right?
# Reservoir weights matrix
reservoir_weight=np.random.normal(0.0, np.sqrt(variance2), size=(num_reservoir_neurons, num_reservoir_neurons))

#Training pass:
#Prepater inputs and targets
import csv
import math
#train_data=[]
#with open('training-set.csv', newline='') as csvfile:
#    csv_val=csv.reader(csvfile)
#    for row in csv_val:
#        train_data.append([float(x) for x in row])
train_data=np.loadtxt('training-set.csv', delimiter=',')
test_data=np.loadtxt('test-set-9.csv', delimiter=',')
train_length=len(train_data)
test_length=len(test_data)

#Initalise reservoir state r_0=0
r=np.zeros((train_length+1, num_reservoir_neurons))

#r.append([0]*500)
print(input_weight[0][0])
print("HI")

#For each t:
for t in range(train_length):
    for i in range(num_reservoir_neurons):
        print(f"t is {t} and i is {i}")
        #print(f" here we have {train_data[t][0]} and {train_data[t][1]} and {train_data[t][2]}")
        #print(reservoir_weight[i][499])
        #print(r[1])
        sum1=sum(reservoir_weight[i][j]*r[t][j] for j in range(num_reservoir_neurons))
        #t=0, i=3
        #print(train_data[t][num_input_neurons])
        #print(input_weight[i])
        sum2=sum(input_weight[i][k]*train_data[t][k] for k in range(num_input_neurons))
        
        r[t+1][i]=(math.tanh(sum1+sum2))
    #r.append(r_local)
    #Update reservoir:
        #r_i(t+1)=tanh(\sum_j w_{ij} r_j (t) + \sum_{k=1}^{N} w_{ik}^{in} x_k(t))
#Drop the first 100 steps
dropped_r=r[101:]
dropped_train=train_data[101:]



#train_data=np.array(train_data)
#r=np.array(r)
#print(f"r_shape is {r.shape}")

W_out=(dropped_train.T@dropped_r) @np.linalg.inv(dropped_r.T @ dropped_r + k_val * np.identity(num_reservoir_neurons))
#Train the readout (ridge_regression)
#X where rows are features (reservoir states, optional input, bias)
#Y: corresponding target outputs
O=np.zeros((train_length, num_output_neurons))
for t in range(train_length):
    for m in range(num_output_neurons):
        #print(W_out[i][num_reservoir_neurons-1])
        #print(r[t])
        O[t][m]=sum(W_out[m][j]*r[t][j] for j in range(num_reservoir_neurons))
        

#Comput readout weights:
    #O_i(t+1)=\sum_j w_{ij}^{out} r_j (t+1)
    #Ridge_parameter k=0.01
    #Use \lambda >0 (ridge)

#predict 500 time steps

r_new=np.zeros((train_length+1, num_reservoir_neurons))

#r.append([0]*500)
print(input_weight[0][0])
print("HI")

#For each t:
for t in range(train_length):
    for i in range(num_reservoir_neurons):
        print(f"t is {t} and i is {i}")
        #print(f" here we have {train_data[t][0]} and {train_data[t][1]} and {train_data[t][2]}")
        #print(reservoir_weight[i][499])
        #print(r[1])
        sum1=sum(reservoir_weight[i][j]*r[t][j] for j in range(num_reservoir_neurons))
        #t=0, i=3
        #print(train_data[t][num_input_neurons])
        #print(input_weight[i])
        sum2=sum(input_weight[i][k]*O[t][k] for k in range(num_input_neurons))
        
        r_new[t+1][i]=(math.tanh(sum1+sum2))


O_new=np.zeros((train_length+test_length+1, num_output_neurons))
O2_new=np.zeros(train_length+test_length+1)
for t in range(train_length):
    for m in range(num_output_neurons):
        #print(W_out[i][num_reservoir_neurons-1])
        #print(r[t])
        O_new[t][m]=sum(W_out[m][j]*r_new[t][j] for j in range(num_reservoir_neurons))
    O2_new[t]=O_new[t][1]

#Calculating new r
output_r=r_new+np.zeros((test_length+1, num_reservoir_neurons)) 


r_final=np.zeros((test_length+1, num_reservoir_neurons))

#r.append([0]*500)
print(input_weight[0][0])
print("HI")

#For each t:
for t in range(test_length):
    for i in range(num_reservoir_neurons):
        print(f"t is {t} and i is {i}")
        #print(f" here we have {train_data[t][0]} and {train_data[t][1]} and {train_data[t][2]}")
        #print(reservoir_weight[i][499])
        #print(r[1])
        sum1=sum(reservoir_weight[i][j]*r_final[t][j] for j in range(num_reservoir_neurons))
        #t=0, i=3
        #print(train_data[t][num_input_neurons])
        #print(input_weight[i])
        sum2=sum(input_weight[i][k]*test_data[t][k] for k in range(num_input_neurons))
        
        r_final[t+1][i]=(math.tanh(sum1+sum2))


for t in range(test_length):
    for m in range(num_output_neurons):
        #print(W_out[i][num_reservoir_neurons-1])
        #print(r[t])
        print(O_new[t+train_length])
        O_new[t+train_length][m]=sum(W_out[m][j]*r_final[t][j] for j in range(num_reservoir_neurons))
    O2_new[t+train_length]=O_new[t+train_length][1]

