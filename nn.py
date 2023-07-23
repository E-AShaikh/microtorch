import random
import numpy as np
from microtorch.engine import Tensor

def Xavier(nInputs):
    return np.sqrt(3.0/nInputs)

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Linear(Module):

    def __init__(self, nInputs, nOutputs, **kwargs):
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        X = Xavier(nInputs)
        self.w = Tensor(np.random.uniform(-X,X,(self.nInputs, self.nOutputs)))
        self.b = Tensor(np.zeros(nOutputs))

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(X)
        out = x @ self.w + self.b
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f'Linear(in_features={self.nInputs}, out_features={self.nOutputs}'

class Sigmoid(Module):

    def __init__(self):
        pass

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x.sigmoid()
    
    def __repr__(self):
        return 'Sigmoid()'

class ReLU(Module):

    def __init__(self):
        pass

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x.relu()

    def __repr__(self):
        return 'ReLU()'

class Softmax(Module):

    def __init__(self):
        pass

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x.softmax()

    def __repr__(self):
        return 'Softmax()'


class CrossEntropyLoss(Module):

    def __init__(self):
        pass
    
    def __call__(self, y, p):
        y = y if isinstance(y, Tensor) else Tensor(y)
        return y.cross_enrtopy_loss(p)

    def __repr__(self):
        return 'CrossEntropyLoss()'
