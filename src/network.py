import torch
from torch import nn
from torchvision import models

class Main_Network(torch.nn.Module):


    def __init__(self):

        super(Main_Network,self).__init__()

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(3,5,kernel_size=5,padding=1,stride=2), #3 input channels for coloured images, 1 for b/w, 5 output
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(5,50,5,2),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU6(),
            torch.nn.Flatten(),
            torch.nn.Linear(9800,1),
            torch.nn.Sigmoid()
            # torch.nn.BatchNorm1d(1),
            # torch.nn.LogSoftmax(dim=1)
        )
    def forward(self, input):
        return self.main(input)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True) #pretrained on resenet, 50 layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1) #fully connected
        self.fc1 = nn.Linear(512 * 4, 256)
        self.fc2 = nn.Linear(512 * 4, 256)
        # self.sigmoid = torch.nn.Sigmoid()

        del self.resnet.fc
        del self.resnet.avgpool

        self.lin_1 = torch.nn.Sequential(
            torch.nn.Linear(256,100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU6(),
        )
        self.lin_2 = torch.nn.Sequential(
            # torch.nn.Linear(101,50*5*5),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x) #batch normalization
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)
        rep = torch.flatten(x,1)
        # x = self.fc(rep)
        x = self.fc1(rep)

        x = self.lin_1(x)
        x = self.lin_2(x)
        return x


class ResnetEncoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.resnet = models.resnet50(pretrained=True) #pretrained on resenet, 50 layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1) #fully connected
        self.fc1 = nn.Linear(512 * 4, latent_dim)
        self.fc2 = nn.Linear(512 * 4, latent_dim)
        # self.sigmoid = torch.nn.Sigmoid()

        del self.resnet.fc
        del self.resnet.avgpool



    def reparametrize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std,device=self.device)
        return mu + std*eps

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x) #batch normalization
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)
        rep = torch.flatten(x,1)
        # x = self.fc(rep)
        mu = self.fc1(rep)
        logvar = self.fc2(rep)

        twelve_sample = False
        z = 0
        if twelve_sample:
            for i in range(12):
                z += self.reparametrize(mu,logvar)
            z /= 12
        else:
            z = self.reparametrize(mu,logvar)
        return z, mu, logvar
    
class Decoder(nn.Module):
    def __init__(self,latent_dim,output_dim):
        super().__init__()

        # self.func = nn.Sequential(
        #     nn.Linear(latent_dim, 1),
        #     nn.Sigmoid()
        # )

        self.lin_1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim,101),
            torch.nn.BatchNorm1d(101),
            torch.nn.ReLU6(),
        )
        self.lin_2 = torch.nn.Sequential(
            # torch.nn.Linear(101,50*5*5),
            torch.nn.Linear(101, 1),
            # torch.nn.Sigmoid()
        )

        self.onelayer = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, output_dim),
            # torch.nn.Sigmoid()
        )

    def forward(self,z):
        # z = self.lin_1(z)
        z = self.onelayer(z)
        # z = self.lin_2(z)
        return z

class FairDecoder(nn.Module):
    def __init__(self,latent_dim,output_dim):
        super().__init__()

        # self.func = nn.Sequential(
        #     nn.Linear(latent_dim+1, 1),
        #     nn.Sigmoid()
        # )

        self.lin_1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+1,100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU6(),
        )
        self.lin_2 = torch.nn.Sequential(
            # torch.nn.Linear(101,50*5*5),
            torch.nn.Linear(101, 1),
            # torch.nn.Sigmoid()
        )

        # self.conv = torch.nn.Sequential(
        #     torch.nn.BatchNorm2d(50),
        #     torch.nn.ReLU6(),
        #     torch.nn.ConvTranspose2d(50,5,5,2),
        #     torch.nn.BatchNorm2d(5),
        #     torch.nn.ReLU6(),
        #     torch.nn.ConvTranspose2d(5,3,5,2,padding=1,output_padding=1),
        #     torch.nn.Sigmoid()
        # )

        self.onelayer = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+1, output_dim),
            # torch.nn.Sigmoid()
        )

    def forward(self, z,a):
        a = a.view(-1,1)
        # z = self.lin_1(torch.cat((z,a),1))
        # z = self.lin_2(torch.cat((z,a),1))


        # z = z.view(-1,50,5,5)
        # output = self.conv(z + a.view((-1,) + (1,) * (len(z.shape) - 1)))

        z = self.onelayer(torch.cat((z,a),1))
        return z
        # return self.func(torch.cat((z,a),dim=1))

class VAE(nn.Module):
    def __init__(self,latent_dim,output_dim=1):
        super().__init__()

        self.encoder = ResnetEncoder(latent_dim)
        self.decoder = Decoder(latent_dim,output_dim)
        self.fair_decoder = FairDecoder(latent_dim,output_dim)

    def forward(self, x, a):
        twelve_sample = False

        if twelve_sample:
            yhat = 0
            yhat_fair = 0
            mu_total = 0
            logvar_total = 0

            for i in range(12):
                z, mu, logvar = self.encoder(x)
                mu_total += mu
                logvar_total += logvar
                yhat += self.decoder(z)
                yhat_fair +=  self.fair_decoder(z,a)
            yhat /= 12
            yhat_fair /= 12
            mu /= 12
            logvar /=12
        else:
            z, mu, logvar = self.encoder(x)
            yhat = self.decoder(z)
            yhat_fair = self.fair_decoder(z, a)

        return yhat, yhat_fair, mu, logvar

    def getz(self,x):
        return self.encoder(x)


class EncoderTabular(torch.nn.Module):

    def __init__(self, latent_dim,input_dim):
        super(EncoderTabular, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fc1 = nn.Linear(100, latent_dim)
        self.fc2 = nn.Linear(100, latent_dim)



        # neural net with single 100 node ReLU layer
        self.func = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 100),
            torch.nn.ReLU(),
        )

        # self.logvar = torch.nn.Parameter(
        #     torch.Tensor([-1.0]))

    def reparametrize(self,mu,logvar):
        std = torch.exp(0.5*logvar)


        eps = torch.randn_like(std,device=self.device)
        return mu + std*eps



    def forward(self, x):
        x = self.func(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparametrize(mu,logvar)
        return z, mu, logvar

# Decoder for the Skoglund I(Z;Y|A) lower bound
class FairDecoderTabular(torch.nn.Module):

    def __init__(self, latent_dim):
        super(FairDecoderTabular, self).__init__()


        self.func = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 1, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),  # output_dim = [1]
        )

    def forward(self, z, a):
        a = a.view(-1,1)
        return self.func(torch.cat((z,a),1))  # 981 x 3 tensor, binary s attribute added to the2 dimensions of Z

# Decoder for the Skoglund I(Z;Y|A) lower bound
class DecoderTabular(torch.nn.Module):

    def __init__(self, latent_dim):
        super(DecoderTabular, self).__init__()


        self.func = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),  # output_dim = [1]
        )

    def forward(self, z):
        return self.func(z)


class VAETabular(torch.nn.Module):

    def __init__(self, latent_dim,input_dim):
        super().__init__()

        self.encoder = EncoderTabular(latent_dim,input_dim)
        self.decoder = DecoderTabular(latent_dim)
        self.fair_decoder = FairDecoderTabular(latent_dim)

    def forward(self, x, a):
        # yhat = 0
        # yhat_fair = 0
        # mu_total = 0
        # logvar_total = 0
        #
        # for i in range(12):
        #     z, mu, logvar = self.encoder(x)
        #     mu_total += mu
        #     logvar_total += logvar
        #     yhat += self.decoder(z)
        #     yhat_fair += self.fair_decoder(z, a)
        # yhat /= 12
        # yhat_fair /= 12
        # mu_total /= 12
        # logvar_total /= 12
        # mu = mu_total
        # logvar = logvar_total

        z, mu, logvar = self.encoder(x)
        yhat = self.decoder(z)
        yhat_fair = self.fair_decoder(z, a)

        return yhat, yhat_fair, mu, logvar
    def getz(self,x):
        return self.encoder(x)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
