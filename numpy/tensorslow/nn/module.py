class Module:
    def forward(self, x):
        pass

    def backward(self):
        pass

    def __call__(self, x):
        return self.forward(x)
