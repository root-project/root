import torch
import torch.nn as nn

#Define model

def generateSequentialModel():
    # Defining the model
    model = nn.Sequential(
           nn.Linear(4,8),
           nn.ReLU(),
           nn.Linear(8,6),
           nn.SELU()
           )

    #Construct loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    #Constructing random test dataset
    x=torch.randn(2,4)
    y=torch.randn(2,6)

    #Training the model
    for i in range(2000):
        y_pred = model(x)
        loss = criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Saving the trained model
    model.eval()
    m = torch.jit.script(model)
    torch.jit.save(m,"PyTorchModelSequential.pt")


def generateModuleModel():
    #Define model
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(6, 36)
            self.fc2 = nn.Linear(36,12)
            self.relu    = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x=self.fc1(x)
            x=self.relu(x)
            x=self.fc2(x)
            x=self.sigmoid(x)
            x=torch.transpose(x,1,0)
            return x

    model = Model()

    #Construct loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    #Constructing random test dataset
    x_train=torch.randn(2,6,requires_grad=True)
    y_train=torch.randn(12,2,requires_grad=True)

    #Training the model
    for i in range(2000):
        y_pred = model(x_train)
        loss = criterion(y_pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Saving the trained model
    model.eval()
    m = torch.jit.script(model)
    torch.jit.save(m,"PyTorchModelModule.pt")


def generateConvolutionModel():
    # Defining the model
    model = nn.Sequential(
                nn.Conv2d(6, 5, 3, stride=2),
                nn.ReLU(),
                )

    #Construct loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    #Constructing random test dataset
    x=torch.randn(5, 6, 5, 5)
    y=torch.randn(5, 5, 2, 2)

    #Training the model
    for i in range(100):
        y_pred = model(x)
        loss = criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Saving the trained model
    model.eval()
    m = torch.jit.script(model)
    torch.jit.save(m,"PyTorchModelConvolution.pt")

generateSequentialModel()
generateModuleModel()
generateConvolutionModel()
