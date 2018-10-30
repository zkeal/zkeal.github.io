---
layout: post
title:  "Logistic Regression"
date:   2018-10-29
excerpt: "The rudimentary algorithm used in NN"
image: "/images/logstic_pic.jpg"
---

## Introduction

In statistics, the logistic model is a widely used basic form model to turn out the logistic function to model a binary dependent variable. In regression 
analysis, logistic regression evaluate the parameters of training data to get its coefficient W.T and b,where it is in the function Y=W.T*X+b which is a form of binomial regression. 
Mathematically, a binary logistic model has a dependent variable with two possible values, such as
pass/fail, win/lose, alive/dead or healthy/sick; these are represented by an indicator variable, where the two values are labeled "0" and "1".

## Method Summary

In the proposed algorithm,Logistic Regression,is extremely similar with classification algorithms because our final predict answers are plenty of labels which
only have two kinds label,'0'and'1',with the possibility to correspond each label. In the logistic model, the the logarithm of the odds for the value 
labeled "1" is a linear combination of one or more independent variables; the independent variables can each be a binary variable or a continuous variable.

__Significant part__

>The logistic function $$g(x)=\frac{1}{1+e^{-x}}$$ 

The graph in python:
> ![image](/images/graph_logistic.jpg)

The logistic function generally help us transfer the result from unlimited to (0,1),the feasibility reflected to label'1'.Therefore,we could get an classification 
from the formula.

## The Implementation

During the process of training, the essential part in the implementation of Logistic Regression is get the min of lost which means the difference between the predict 
data and label.That is using derivative dz=a[i]-y[i] to time X so that if the derivative is closer to 0,it will get less distance in movement. 

The sum of Lost Function
> ![image](/images/logistic_lose.jpg)

__the overview of algorithm__
<center>
<img src="/images/LR.png" width="75%" height="75%" />
</center>

__gradient derivative__

```
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    ### END CODE HERE ###
    cost = np.squeeze(cost)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    grads = {"dw": dw,
             "db": db}

    return grads, cost
```

__optimize__

```
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

```

After these,the only thing is to predict the test data by using the calculated W and b.

## SUMMARY
This algorithm is simple but significant because it is not only used as the cornerstone in
neural networks,but also a splendid example for machine learning.
