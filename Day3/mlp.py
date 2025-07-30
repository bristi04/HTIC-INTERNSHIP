import numpy as np

x=np.array([[1],[4],[2]])
y=np.array([[1]])

input_size=len(x)
hidden_size=2
output_size=1
lr=0.01

w1=np.random.randn(hidden_size,input_size)
b1=np.zeros((hidden_size,1))

w2=np.random.randn(output_size,hidden_size)
b2=np.zeros((output_size,1))

def relu(z):
    return np.maximum(0,z)

def relu_deriv(z):
    return (z>0).astype(float)

#Forward Propagation
for i in range(1000):
    h1=np.dot(w1,x)+b1
    a1=relu(h1)

    h2=np.dot(w2,a1)+b2
    y_pred=h2

    loss=0.5*(y_pred-y)**2
    
    #Back Propagation
    dh2 = y_pred - y
    dw2 = dh2 @ a1.T  
    db2 = dh2

    da1 = w2.T @ dh2
    dh1 = da1 * relu_deriv(h1)
    dw1 = dh1 @ x.T
    db1 = dh1


    w2=w2-lr*dw2
    w1=w1-lr*dw1

    b1 = b1 - lr * db1
    b2 = b2 - lr * db2


    if i % 100 == 0:
        print(f"Step {i}-Loss: {loss.item():.4f}")
