#ACTIVATION FUNCTIONS
import math 
import matplotlib.pyplot as plt
import numpy as np


#Sigmoid Function
def sigmoid(x):
  res=1/(1+np.exp(-x))
  return res

x=np.linspace(-10,10,100)
y=sigmoid(x)
plt.subplot(1,4,1)
plt.stem(x,y)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)


#Tanh Function
def tanh(x):
  res=2/(1+np.exp(-2*x))
  return res

x=np.linspace(-10,10,100)
y=tanh(x)
plt.subplot(1,4,2)
plt.stem(x,y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.title('tanh function')

#ReLU FUNCTION

def relu(x):
    res=np.max(0,x)
    return res


x=np.linspace(-10,10,100)
y=relu(x)
plt.subplot(1,4,3)
plt.stem(x,y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.title('ReLU function')

plt.tight_layout()
plt.show()


#SOFTMAX FUNCTION
def softmax(x):
    res=np.exp(x) / np.sum(np.exp(x), axis=0)
    return res

x=np.linspace(-10,10,100)
y=softmax(x)
plt.subplot(1,4,4)
plt.stem(x,y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('softmax(x)')
plt.title('Softmax function')

plt.tight_layout()
plt.show()
