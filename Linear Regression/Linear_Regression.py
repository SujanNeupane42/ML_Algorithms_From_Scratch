import numpy

class LinearRegression:
    def __init__(self): 
        self.lr = 0.001
        self.n_iters = 1000 # 30000
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        
        n_samples, n_features  = X.shape
        self.weights = numpy.zeros(n_features)
        self.bias = 0
        
        for i in range(0, self.n_iters):
            y_pred = numpy.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * numpy.dot(X.T, (y_pred - y))
            db = (1/n_samples) * numpy.sum(y_pred - y)
            
            self.weights -= (self.lr * dw)
            self.bias  -= (self.lr * db)
    
    def predict(self, X):
        y_pred = numpy.dot(X, self.weights) + self.bias
        return y_pred

X = numpy.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])
y = numpy.array([100,200,300,400,500])

lin_reg = LinearRegression()
lin_reg.fit(X,y)
predictions = lin_reg.predict(X)
print(predictions)
