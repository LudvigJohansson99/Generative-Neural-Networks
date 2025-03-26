import sys
import os
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import torch.optim

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dataloader_path = os.path.abspath("/home/ossian/Datasets/celeba/dataloader.py")
spec = importlib.util.spec_from_file_location("dataloader", dataloader_path)
celeba = importlib.util.module_from_spec(spec)
sys.modules["dataloader"] = celeba
spec.loader.exec_module(celeba)





dictionary = torch.load("model_weights_DCGAN.pth")

filtered_model = {k: v for k, v in dictionary.items() if "decoder" in k}

print(filtered_model.keys())



BCELoss = nn.BCEWithLogitsLoss()


class Discriminator(nn.Module):
    def __init__(self, device="cpu"):
        super(Discriminator, self).__init__()

        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 124x124x3 -> 62x62x16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 62x62x16 -> 31x31x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 31x31x32 -> 16x16x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16x64 -> 8x8x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Flatten()  # Flatten for fully connected layer
        )
        self.linear_layer = nn.Linear(8*8*128, 1)

    def forward(self, x):
        return self.linear_layer(self.encoder(x))



class Generator(nn.Module):
    def __init__(self, input_dimension=128, device="cpu"):
        super(Generator, self).__init__()

        self.device = device
        self.input_dimension = input_dimension

        self.decoder_fc = nn.Linear(input_dimension, 8*8*128)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),  # Unflatten for deconvolutional layers
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x128 -> 16x16x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x64 -> 31x31x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 31x31x32 -> 62x62x16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 62x62x16 -> 124x124x3
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        self.decoder_fc = nn.Linear(self.input_dimension, 8*8*128)

    def forward(self, n):
        z = torch.randn((n, self.input_dimension), device=self.device)
        return self.decoder(self.decoder_fc(z))



nr_epochs = 10
batch_size = 32
gen_labels = torch.zeros(batch_size, device=device)
true_labels = torch.ones(batch_size, device=device)
labels = torch.cat((gen_labels, true_labels),dim=0)
labels = labels.view(batch_size*2,1)

gen = Generator(device=device)
gen.load_state_dict(filtered_model, strict=False)
disc = Discriminator(device=device)

gen.to(device)
disc.to(device)

disc_optimizer = optim.SGD(disc.parameters(), lr=1e-4)
gen_optimizer = optim.Adam(gen.parameters(), lr=1e-6)

count = 0
#gen_loss_cap = 0.5
disc_loss_cap = 0.2
disc_losses = []
gen_losses = []
generator_count = 0
discriminator_count = 0
for e in range(nr_epochs):
    samples = gen(64)
    samples = (samples + 1)/2
    samples.to("cpu")
    grid_img = torchvision.utils.make_grid(samples, nrow=8)
    grid_img_pil = torchvision.transforms.ToPILImage()(grid_img)
    grid_img_pil.save("GAN_pretrained_with_VAE" + str(e) + ".png")
    for img in celeba.dataloader:

        if img.shape[0] != batch_size:
            break
        img = img*2-1
        img = img.to(device)
        generated_data = gen(batch_size)



        disc_data = torch.cat((generated_data, img), dim=0)


        logits = disc(disc_data)


        disc_loss = BCELoss(logits, labels)
        disc_losses.append(disc_loss.item())
        gen_loss = BCELoss(logits[:batch_size, 0], true_labels)
        gen_losses.append(gen_loss.item())
        if disc_loss.item() > disc_loss_cap:
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            discriminator_count += 1
        else:
            gen_optimizer.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=0.5)
            gen_optimizer.step()
            generator_count += 1
        count += 1
    count = 0
    print("trained generator: ", generator_count)
    print("trained discriminator: ", discriminator_count)
    generator_count = 0
    discriminator_count = 0


samples = gen(64)
samples.to("cpu")
samples = (samples + 1)/2
grid_img = torchvision.utils.make_grid(samples, nrow=8)
grid_img_pil = torchvision.transforms.ToPILImage()(grid_img)
grid_img_pil.save("GAN_pretrained_with_VAE" + str(nr_epochs) + ".png")




plt.plot(disc_losses)
plt.plot(gen_losses)
plt.show()
#vae = ConvVAE(128)
#optimizer = Adam(vae.parameters(), lr=1e-3)
#
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#vae.to(device)
## Training loop
#num_epochs = 10
#
#for epoch in range(num_epochs):
#    train_loss = 0
#    samples = vae.generate(64)
#
#    grid_img = torchvision.utils.make_grid(samples, nrow=8)
#    grid_img_pil = torchvision.transforms.ToPILImage()(grid_img)
#    grid_img_pil.save("Conv_Uppsample_Epoch_" + str(epoch) + ".png")
#
#    for img in celeba.dataloader:
#        img = img.to('cuda')
#        recon, mu, logvar = vae(img)
#        loss = vae_loss(recon, img, mu, logvar, 1e-5)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        train_loss += loss.item()
#
#    print(f'====> Epoch {epoch} Average loss: {train_loss / len(celeba.dataloader.dataset)}')
#
#
#grid_img = torchvision.utils.make_grid(samples, nrow=8)
#plt.figure(figsize=(5, 1))
#plt.imshow(grid_img.permute(1, 2, 0))
#plt.axis('off')
#plt.show()
#torch.save(vae.state_dict(), "model_weights.pth")
