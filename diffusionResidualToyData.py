import os
import sys 
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

module_path = os.path.abspath("/home/ossian/Datasets/2dToyData/")
if module_path not in sys.path:
    sys.path.append(module_path)

import toyData as tD

def loadData(data, batch_size=4, shuffle=True):
    if data[1] is None:
        dataset = TensorDataset(data[0])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), data[0]

    dataset = TensorDataset(*data)
    return DataLoader(dataset, batch_size=4, shuffle=True), data[0]

class MLP(nn.Module):
    def __init__(self, input_size=2, output_size=2, output_func=None, hidden=[4,8,16,8,4], device="cpu"):
        super(MLP, self).__init__()
        layers = []
        for h in hidden:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.LeakyReLU())
            input_size = h
        layers.append(nn.Linear(input_size, output_size))
        if not output_func is None:
            layers.append(output_func)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Diffusion(MLP):
    """ Diffusion Model that uses a simple MLP, with time encoding that is concatinated with the input """
    def __init__(self, beta_start=0.02, beta_end=1e-3, time_steps=50, output_size=2, output_func=None, hidden=[4,8,16,8,4], device="cpu"):
        super(Diffusion, self).__init__(output_size + 1, output_size, output_func, hidden, device)
        self.device=device
        self.output_size = output_size
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps
        self.scheduler()

    def scheduler(self):
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.time_steps, device=self.device)
        self.alpha =  1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        

    def add_noise(self, x, t):
        noise = torch.randn(x.shape, device=self.device)
        return torch.sqrt(self.alpha[t])*x + torch.sqrt(1 - self.alpha[t] )*noise, noise 

    def predict_noise(self, x, t):
        return x - super().forward(torch.cat((x,t.to(torch.float).repeat(x.shape[0]).view(x.shape[0],1)),dim=1))

    def forward(self, nr_samples):
        x = torch.randn((nr_samples, self.output_size), device=self.device)
        time_steps = torch.linspace(self.time_steps-1, 0, self.time_steps, device=self.device)
        print(time_steps)
        for t in time_steps:
            predicted_noise = self.predict_noise(x, t)

            mean = (x - self.beta[t.to(torch.int)]/torch.sqrt(1-self.alpha_hat[t.to(torch.int)])*predicted_noise)/torch.sqrt(self.alpha[t.to(torch.int)])
            if t == 0:
                x = mean
            else:
                z = torch.randn((nr_samples, self.output_size), device=self.device)
                x = mean + torch.sqrt(self.beta[t.to(torch.int)])*z
        return x





# Check if CUDA is available
cuda_available = torch.cuda.is_available()

print(f"CUDA available: {cuda_available}")
batch_size = 4
dataloader, data = loadData(tD.generate_data("spiral", device="cuda", plot=False), batch_size=batch_size)

model = Diffusion(device="cuda")
model.to("cuda")

nr_epochs = 10

optimizer = optim.Adam(model.parameters(), lr=1e-3)

count = 0
losses = []
mse_loss = nn.MSELoss()
for e in range(nr_epochs):
    for d in dataloader:


        t = torch.randint(0, model.time_steps, (1,), device=model.device)
        
        noised_data, noise = model.add_noise(d[0], t)

        predicted_noise = model.predict_noise(noised_data, t)
        loss = mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()


        





plt.figure()
np_data = data.detach().to("cpu").numpy()
x = np_data[:, 0]
y = np_data[:, 1]
plt.scatter(x,y,s=7)
data = model(1000)
np_data = data.detach().to("cpu").numpy()
x = np_data[:, 0]
y = np_data[:, 1]
plt.scatter(x,y,s=7)
#plt.scatter(2.5,2.5,s=7)




        

        

plt.show()


plt.plot(losses)
plt.show()
