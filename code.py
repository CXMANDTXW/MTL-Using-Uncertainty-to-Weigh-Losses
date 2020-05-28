import torch
import torch.nn as nn
import numpy as np
np.random.seed(0)

class layer(nn.Module):
    def __init__(self):
        super(layer,self).__init__()

        class Mtl(nn.Module):
            def __init__(self):
                super(Mtl, self).__init__()
                self.sigma1 = nn.Parameter(torch.zeros(1))
                self.sigma2 = nn.Parameter(torch.zeros(1))
                self.log_vars = [self.sigma1,self.sigma2]

            def forward(self,inputs):
                ys_true = inputs[:2]
                ys_pred = inputs[2:]
                # log_vars = [nn.parameter(torch.zeros(1)),nn.parameter(torch.zeros(1))]
                loss=0
                for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
                    pre = torch.exp(-log_var)
                    loss += torch.sum(pre*(y_true-y_pred)**2+log_var,-1)
                loss = torch.mean(loss)
                return loss
        self.linear1 = nn.Sequential(
            nn.Linear(1,1024),
            nn.ReLU()
        )
        self.y1 = nn.Linear(1024,1)
        self.y2 = nn.Linear(1024,1)
        self.out = Mtl()
    def forward(self,X,Y1,Y2):
        x=self.linear1(X)
        y_1=self.y1(x)
        y_2=self.y2(x)
        out=self.out([Y1,Y2,y_1,y_2])
        return out


N = 100
nb_epoch = 5000
batch_size = 20
nb_features = 1024
Q = 1
D1 = 1  # first output
D2 = 1  # second output
def gen_data(N):
    X = np.random.randn(N, Q)
    w1 = 2.
    b1 = 8.
    sigma1 = 1e1  # ground truth
    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    w2 = 3
    b2 = 3.
    sigma2 = 1e0  # ground truth
    Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2
X, Y1, Y2 = gen_data(N)
X=torch.from_numpy(X).type(torch.FloatTensor)
Y1=torch.from_numpy(Y1).type(torch.FloatTensor)
Y2=torch.from_numpy(Y2).type(torch.FloatTensor)
model=layer()

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
for i in range(nb_epoch):
    tmp=0
    step=0
    loss=model(X,Y1,Y2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%100==0:
        print("At epoch %d, Loss: %.4f" % (i, loss))
        print('sigma1: ',model.state_dict()['out.sigma1'])
        print('sigma2: ',model.state_dict()['out.sigma2'])
sigma1 = model.state_dict()['out.sigma1']
sigma2 = model.state_dict()['out.sigma2']
print(torch.exp(sigma1)**0.5)
print(torch.exp(sigma2)**0.5)

