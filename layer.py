import numpy as np
class linear:
    def __init__(self, in_dim, out_dim, W=None, b=None,):
        if W is None and b is None:
            self.W = np.random.randn(out_dim, in_dim)*0.01
            self.b =  np.zeros((out_dim, 1))
        else:
            self.W = W
            self.b = b
        self.X_in = None
        self.grad_W = None
        self.grad_b = None
    def forward(self, X_in):
        self.X_in = X_in
        return self.W@X_in + self.b
    def backward(self, G_out):
        G_in = self.W.T@G_out
        self.grad_W = (G_out@self.X_in.T)/G_out.shape[1]
        self.grad_b = np.mean(G_out, axis=1, keepdims=True)
        return G_in
    def update(self, a):
        self.W-=a*self.grad_W
        self.b-=a*self.grad_b
    def save_Weight(self):
        return self.W, self.b

class relu:
    def __init__(self):
        self.RX_out = None
    def forward(self, X_out):
        self.RX_out = X_out > 0
        return np.maximum(0, X_out)
    def backward(self,G_in):
        return self.RX_out*G_in
    def update(self, a):
        pass

            
    

        