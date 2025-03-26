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

class Generator(MLP):
    def __init__(self, input_size=2, output_size=2, output_func=None, hidden=[4,8,16,8,4], device="cpu"):
        super(Generator, self).__init__(input_size, output_size, output_func, hidden, device)
        self.BCELoss = nn.BCEWithLogitsLoss()
    def Loss(self, logit, label):
        return self.BCELoss(logit, label)


        

class Discriminator(MLP):
    def __init__(self, input_size=2, hidden=[4,64,32,8,4], device="cpu"):
        super(Discriminator, self).__init__(input_size, 1, None, hidden, device)
        self.BCELoss = nn.BCEWithLogitsLoss()
    def Loss(self, logit, label):
        return self.BCELoss(logit, label)


def plot_decision_boundary(model, device="cpu"):
    # Generate a grid of points
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Flatten the grid to pass into the model
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    # Get predictions for each point in the grid
    with torch.no_grad():
        predictions = model(grid_points)

    # Reshape the predictions to match the grid shape
    Z = predictions.to("cpu").numpy().reshape(xx.shape)

    # Plot the decision boundary
    contour= plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    cbar = plt.colorbar(contour, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
    plt.ylabel('Feature 2')



# Check if CUDA is available
cuda_available = torch.cuda.is_available()

print(f"CUDA available: {cuda_available}")
batch_size = 4
dataloader, data = loadData(tD.generate_data("spiral", device="cuda", plot=False), batch_size=batch_size)


disc = Discriminator()
gen = Generator()
disc.to("cuda")
gen.to("cuda")



nr_epochs = 100
gen_labels = torch.zeros(batch_size, device="cuda")
true_labels = torch.ones(batch_size, device="cuda")
labels = torch.cat((gen_labels, true_labels),dim=0)
labels = labels.view(batch_size*2,1)

disc_optimizer = optim.Adam(disc.parameters(), lr=1e-3)
gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3)

count = 0
gen_loss_cap = 0.5
disc_loss_cap = 0.5
disc_losses = []
gen_losses = []
for e in range(nr_epochs):
    for d in dataloader:
        #rand_indicies = torch.randperm(labels.size(0))


        noise = torch.randn(batch_size, 2, device="cuda")
        generated_data = gen(noise)



        disc_data = torch.cat((generated_data, d[0]), dim=0)

        #shuffled_data = disc_data[rand_indicies]

        logits = disc(disc_data)

        #shuffled_labels = labels[rand_indicies]

        disc_loss = disc.Loss(logits, labels)
        disc_losses.append(disc_loss.item())
        gen_loss = gen.Loss(logits[:batch_size, 0], true_labels)
        gen_losses.append(gen_loss.item())
        if disc_loss.item() > disc_loss_cap:
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
        else:
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()


plt.figure()
np_data = data.detach().to("cpu").numpy()
x = np_data[:, 0]
y = np_data[:, 1]
plt.scatter(x,y,s=7)
data = gen(torch.randn(1000, 2, device="cuda"))
np_data = data.detach().to("cpu").numpy()
x = np_data[:, 0]
y = np_data[:, 1]
plt.scatter(x,y,s=7)
#plt.scatter(2.5,2.5,s=7)




        

# Plot the decision boundary
plot_decision_boundary(disc, "cuda")
        

plt.show()


plt.plot(disc_losses)
plt.plot(gen_losses)
plt.show()
