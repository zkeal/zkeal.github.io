---
layout: post
title:  "Neural Network"
date:   2019-01-07
excerpt: "The Fundamental Neural Network"
image: "/images/F_NN.jpg"
---

## Introduce Netrual Networks

According to WiKi, the Neural Networks (NN) or connection systems are computing systems vaguely inspired by the biological 
neural networks that constitute animal brains. Essentially, however,I think the Neural Network is just a kind of Hash algorithm, 
which compresses the pieces of data, subsequently compare with the compressing result(Hash Value) to get an consequences. But one
special point is that comparing approach is not a equal but seeking min cost by an Gradient Descent algorithm.
   
## Principles

In brief, the procedure is familiar with other machine learning algorithm, constructing the layers and nodes(determine how many
layers and nodes by **man**), updating parameters by using labeled data, and discipline data by model.

_There is an example of Neural Network, one hidden layers and its core is Relu function, and the core of output layer is sigmoid function_
<p align="center">
    <img src="/images/FandB.png" width="80%">
</p>


__Step1:__
init parameters
In the equation Y=Wx+B, the W can not be inited as 0, otherwise the result can't change, but B=0 does not matter.
```
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
```
assemble parameters
Notice that all parameters mean matrix
```
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    for l in range(1, L):
    
        parameters['W' + str(l)] = np.random.rand(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

```
__Step2:__
Linear-Activation Linear-Activation
Use the label and previous parameters to calculate the result, then compare the label and computing result.
```
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
```
Implement the whole Forward activation

```
for l in range(1, L):
        A_prev = A 
       
        W_L=parameters['W' + str(l)]
        B_L=parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev,W_L,B_L,activation = "relu")
        caches.append(cache)
 
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    W=parameters['W'+str(L)]
    b=parameters['b'+str(L)]
    AL, cache = linear_activation_forward(A,W,b,activation = "sigmoid")
    caches.append(cache)

```

__Step3:__
Cost function
```
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    m  =  Y . shape [ 1 ]
    cost = -(np.sum(np.multiply(Y,np.log(AL)))+np.sum(np.multiply(1-Y,1-AL)))/m
```
Cost function is used for checking the discipline. Usually use (plt.plot(np.squeeze(cost))) to observe the result.

__Step4:__
Backward
The essence of Backward is using the result to adjust previous parameters.

_This is a interpretation of Backward, Z means the Forward result._
<div align="center">
    <img src="/images/Back_linear.png" align="center">
</div>


```
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
```

__Step5:__
Finally Update parameters
```
for l in range(L):
        parameters["W" + str(l+1)] =parameters["W" + str(l+1)] - learning_rate * grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db"+ str(l+1)]
```

This the main procedure of Neural Network. After constant calculating times, the loss will optimize at min value.
Eventually, use the model to predict the data set by Forward activation.
## SUMMARY

This Neural Network algorithm model is the simplest, but its frame is very clear. Furthermore, this frame can extend by 
changing the quantity of hidden layers, using other core functions, and differing cost functions. These steps still need
people to take part.
