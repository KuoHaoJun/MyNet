'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-05-12 13:23:49
@LastEditTime: 2019-09-01 21:46:51
@LastEditors: Please set LastEditors
'''
import numpy as np  
def affine_forward(x, w, b):   
    """    
    Computes the forward pass for an affine (fully-connected) layer. 
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N   
    examples, where each example x[i] has shape (d_1, ..., d_k). We will    
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and    
    then transform it to an output vector of dimension M.    
    Inputs:    
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)    
    - w: A numpy array of weights, of shape (D, M)    
    - b: A numpy array of biases, of shape (M,)   
    Returns a tuple of:    
    - out: output, of shape (N, M)    
    - cache: (x, w, b)   
    """
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)         # (N,D)
    out = np.dot(x_row, w) + b       # (N,M)
    cache = (x, w, b)

    return out,cache

def affine_backward(dout, cache):   
    """    
    Computes the backward pass for an affine layer.    
    Inputs:    
    - dout: Upstream derivative, of shape (N, M)    
    - cache: Tuple of: 
    - x: Input data, of shape (N, d_1, ... d_k)    
    - w: Weights, of shape (D, M)    
    Returns a tuple of:   
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)    
    - dw: Gradient with respect to w, of shape (D, M) 
    - db: Gradient with respect to b, of shape (M,)    
    """    
    x, w, b = cache    
    dx, dw, db = None, None, None   
    dx = np.dot(dout, w.T)                       # (N,D)    
    dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)   
    x_row = x.reshape(x.shape[0], -1)            # (N,D)    
    dw = np.dot(x_row.T, dout)                   # (D,M)    
    db = np.sum(dout, axis=0, keepdims=True)     # (1,M)    

    return dx, dw, db
#输入
X = np.array([[1,1],  
            [-1,1],  
            [-1,-1],  
            [1,-1]]) 
t = np.array([0,1,2,3])  
np.random.seed(1)  #可以去掉
print (X.shape[1])
# 一些初始化参数
weight_scale = 1
input_dim = X.shape[1]     #输入维度
num_classes = t.shape[0]   #输出维度
hidden_dim = 50             #隐藏层维度
reg = 0.001
epsilon = 0.001
# randomly initialize our weights with mean 0  
W1 = weight_scale * np.random.randn(input_dim, hidden_dim)  
W2 = weight_scale * np.random.randn(hidden_dim, num_classes)  
b1 = np.zeros((1, hidden_dim))
b2 = np.zeros((1, num_classes))

# 训练循环
for j in range(10000):
    # 前向传播
    H,fc_cache = affine_forward(X,W1,b1)  #仿射
    H = np.maximum(0, H) #激活
    relu_cache = H
    Y,cachey = affine_forward(H,W2,b2)  #仿射
    
    # Softmax
    probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))    
    probs /= np.sum(probs, axis=1, keepdims=True)  # Softmax
    
    N = Y.shape[0]  
    print(probs[np.arange(N), t])
    loss = -np.sum(np.log(probs[np.arange(N), t])) / N
    print(loss)

    # 反向传播
    dx = probs.copy()    
    dx[np.arange(N), t] -= 1    
    dx /= N    #到这里是反向传播到softmax前
    # Backward pass: compute gradients
    dh1, dW2, db2 = affine_backward(dx, cachey)

    dh1[relu_cache <= 0] = 0 
    dX, dW1, db1 = affine_backward(dh1, fc_cache)
    dW2 += reg * W2
    dW1 += reg * W1

    W2 += -epsilon * dW2
    b2 += -epsilon * db2
    W1 += -epsilon * dW1
    b1 += -epsilon * db1

test = np.array([[2,2],[-2,2],[-2,-2],[2,-2]])
H,fc_cache = affine_forward(test,W1,b1)  #仿射
H = np.maximum(0, H) #激活
relu_cache = H
Y,cachey = affine_forward(H,W2,b2)  #仿射
    # Softmax
probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))    
probs /= np.sum(probs, axis=1, keepdims=True)  # Softmax
for k in range(4):
    print(test[k,:],"所在的象限为",np.argmax(probs[k,:])+1)
print(probs)