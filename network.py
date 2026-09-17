import numpy as np
import layer as l
class MLP:
    def __init__(self, Ws=None, bs=None):
        if Ws  is None and bs is None:
            self.layers = [l.linear(784, 256), l.relu(), l.linear(256, 128), l.relu(), l.linear(128, 10)]
        else:
            self.layers = [l.linear(784, 256, Ws[0], bs[0]), l.relu(), l.linear(256, 128, Ws[1], bs[1]), l.relu(), l.linear(128, 10, Ws[2], bs[2])]
        
        self.X_out = None
    def forward(self, X_in):
        for layer in self.layers:
            X_in = layer.forward(X_in)
        self.X_out = X_in
    def backward(self, G_out):
        for layer in reversed(self.layers):
            G_out = layer.backward(G_out)
    def update(self, a):
        for layer in self.layers:
            layer.update(a)
    def softmax(self, x):
        e_x = np.exp(x)
        return e_x/e_x.sum(axis=0)
    def predict(self, X_in):
        self.forward(X_in)
        p = self.softmax(self.X_out)
        return np.argmax(p)
    def loss(self, correct):
        p = self.softmax(self.X_out)
        L = np.empty(correct.size)
        one_hot = np.zeros(p.shape)
        for j in range(0, correct.size):
            L[j] = -np.log(p[correct[j], j])
            one_hot[correct[j], j] = 1
            
        J = L.mean()
        G_out = p - one_hot
        return J, G_out
    def save_weight(self):
        Ws = []
        bs = []
        for layer in self.layers:
            if isinstance(layer, l.linear):
                W, b = layer.save_Weight()
                Ws.append(W)
                bs.append(b)
        return Ws, bs



