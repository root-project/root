import torch
import torch.nn as nn

#Define model
model = nn.Sequential(
           nn.Linear(4,6),
           nn.ReLU()
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
