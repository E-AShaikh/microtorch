import numpy

class Tensor:
    

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    @property
    def shape(self):
        return self.data.shape

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def dim(self):
        return self.data.ndim

    def zero_grad(self):
        self._grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.matmul(out.grad, other.data.transpose())
            other.grad += self.data[:,:,np.newaxis] * out.grad[:,np.newaxis,:]
        out._backward = _backward

        return out

    def sigmoid(self):
        bounded = np.maximum(-10, np.minimum(10, self.data)) # blocks numerical warnings
        out = Tensor(1. / 1 + np.exp(-bounded), (self), 'Sigmoid')

        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def softmax(self):
        smax = np.max(self.data, axis=1, keepdims=True)
        bounded = np.maximum(-10,self.s.value - smax) # blocks numerical warnings
        es = np.exp(bounded)
        out = Tensor(es / np.sum(es, axis=1, keepdims=True), (self), 'Softmax')

        def _backward():
            p_dot_pgrad = np.matmul(out.data[:,np.newaxis,:], out.grad[:,:,np.newaxis]).squeeze(-1) # p dot p.grad with shape (nBatch,1)
            self.grad += out.data * (out.grad - p_dot_pgrad) # p_dot_pgrad is broadcast over the labels
        out._backward = _backward

        return out

    def cross_entropy_loss(self, p):
        b = len(self.data)
        neg_log = -np.log(p)
        out = Tensor(np.sum(y * neg_log) / b, (self, p), 'CrossEntropyLoss')

        def _backward():
            self.grad += out.grad * - (self.data / p.data)
            p.grad += out.grad * - neg_log
        out._backward = _backward

        return out


    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    
    
  

