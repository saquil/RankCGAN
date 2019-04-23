import utils, torch, time, os, pickle, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from dataloader import dataloader
from dataloader import pairloader

class discriminator(nn.Module):
    # Network Architecture is exactly same as in DCGAN
    def __init__(self, input_dim=3, output_dim=1, input_size=64, rank_code=1):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size        
        

        self.nconvs = int(math.log2(self.input_size)) - 3
        mult = 1
        layers = []

         
        layers.append(nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ))
        
        for i in range(0, self.nconvs):
            layers.append(nn.Sequential(
            nn.Conv2d(64 * mult, 64 * mult * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * mult * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ))
            mult = mult * 2
       
        layers.append(nn.Sequential(
            nn.Conv2d(mult * 64, 1, 4, bias=False),
            nn.Sigmoid(),
        ))
        self.main = nn.Sequential(*layers)

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.main(input)
        x = x.view([-1])
        return x

class generator(nn.Module):
    # Network Architecture is exactly same as in DCGAN
    def __init__(self, input_dim=100, output_dim=1, input_size=64, rank_code=1):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.rank_code = rank_code  # rank latent variable

        self.nconvs = int(math.log2(self.input_size)) - 3
        mult = self.input_size // 8
        layers = []
      
       
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(self.input_dim + self.rank_code, 64 * mult, 4, bias=False),
            nn.BatchNorm2d(64 * mult),
            nn.ReLU(True),
            ))
        
        for i in range(self.nconvs):
            layers.append(nn.Sequential(
            nn.ConvTranspose2d(64 * mult, 64 * (mult//2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * (mult//2)),
            nn.ReLU(True),
            ))
            mult = mult // 2

        layers.append(nn.Sequential(
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1, bias=False),
            nn.Tanh(),
            ))
        self.main = nn.Sequential(*layers)
        
        utils.initialize_weights(self)


    def forward(self, input, rank_code):
        x = torch.cat([input, rank_code], 1)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.main(x)
        return x

    
class encoder_r(nn.Module):
    # Network Architecture is exactly same as in DCGAN
    def __init__(self, input_dim=3, output_dim=1, input_size=64, rank_code=1):
        super(encoder_r, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size        
        

        self.nconvs = int(math.log2(self.input_size)) - 3
        mult = 1
        layers = []

         
        layers.append(nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ))
        
        for i in range(0, self.nconvs):
            layers.append(nn.Sequential(
            nn.Conv2d(64 * mult, 64 * mult * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * mult * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ))
            mult = mult * 2
       
        layers.append(nn.Sequential(
            nn.Conv2d(mult * 64, output_dim, 4, bias=False),
            nn.Sigmoid(),
        ))
        self.main = nn.Sequential(*layers)

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.main(input)
        x = x.view([-1, self.output_dim])
        # transform to [-1,1]
        x = 2*x-1
        return x
    
class encoder_z(nn.Module):
    # Network Architecture is exactly same as in DCGAN
    def __init__(self, input_dim=3, output_dim=1, input_size=64, rank_code=1):
        super(encoder_z, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size        
        

        self.nconvs = int(math.log2(self.input_size)) - 3
        mult = 1
        layers = []

         
        layers.append(nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ))
        
        for i in range(0, self.nconvs):
            layers.append(nn.Sequential(
            nn.Conv2d(64 * mult, 64 * mult * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * mult * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ))
            mult = mult * 2
       
        layers.append(nn.Sequential(
            nn.Conv2d(mult * 64, output_dim, 4, bias=False),
        ))
        self.main = nn.Sequential(*layers)

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.main(input)
        x = x.view([-1, self.output_dim])
        return x



class encoder(object):
    def __init__(self, args, SUPERVISED=True):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = 'RankCGAN'
        self.input_size = args.input_size
        self.z_dim = 100
        self.rank_code = 1         # rank latent variable
        self.sample_num = 100

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        self.pair_loader = pairloader(self.dataset, 'sporty', self.input_size, self.batch_size//2)
        data = self.data_loader.__iter__().__next__()
        pair = self.pair_loader.__iter__().__next__()
        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, rank_code=self.rank_code)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, rank_code=self.rank_code)
        self.Ez = encoder_z(input_dim=data.shape[1], output_dim=100, input_size=self.input_size, rank_code=self.rank_code)
        self.Er = encoder_r(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, rank_code=self.rank_code)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.Ez_optimizer = optim.Adam(self.Ez.parameters(), lr=args.lrEz, betas=(args.beta1, args.beta2))
        self.Er_optimizer = optim.Adam(self.Er.parameters(), lr=args.lrEr, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.Ez.cuda()
            self.Er.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.Ez)
        utils.print_network(self.Er)
        print('-----------------------------------------------')

        # fixed noise & condition

        lines = int(np.floor(np.sqrt(self.sample_num)))
        self.sample_z = torch.zeros((self.sample_num, self.z_dim))
        for i in range(lines):
            self.sample_z[i * lines] = torch.randn(1, self.z_dim)
            for j in range(1, lines):
                self.sample_z[i * lines + j] = self.sample_z[i * lines]
                
        
        self.sample_y = torch.linspace(-1,1,lines).expand(lines, lines).contiguous().view([-1,1])
        

        if self.gpu_mode:
            self.sample_z, self.sample_y = self.sample_z.cuda(), self.sample_y.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['Ez_loss'] = []
        self.train_hist['Er_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.Ez.train()
        self.Er.train()
        self.G.eval()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for iter, x_ in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.randn((self.batch_size, self.z_dim))
                y_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size,1))).type(torch.FloatTensor)
                
                if self.gpu_mode:
                    x_, z_, y_ = x_.cuda(), z_.cuda(), y_.cuda()
               
                # update Ez network
                self.Ez_optimizer.zero_grad()

                G_ = self.G(z_, y_)
                z_e = self.Ez(G_.detach())

                Ez_loss = self.MSE_loss(z_e,z_.detach())

                self.train_hist['Ez_loss'].append(Ez_loss.item())

                Ez_loss.backward()
                self.Ez_optimizer.step()
                
                # update Er network
                self.Er_optimizer.zero_grad()  
                
                y_e = self.Er(G_.detach())
                
                Er_loss = self.MSE_loss(y_e,y_.detach())  
                
                self.train_hist['Er_loss'].append(Er_loss.item())
                
                Er_loss.backward()
                self.Er_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] Er_loss: %.8f, Ez_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, Er_loss.item(), Ez_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            #with torch.no_grad():
            #    self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        """ manipulating one continous code """
        samples = self.G(self.sample_z, self.sample_y)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.Ez.state_dict(), os.path.join(save_dir, self.model_name + '_Ez.pkl'))
        torch.save(self.Er.state_dict(), os.path.join(save_dir, self.model_name + '_Er.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        
        if os.path.exists(os.path.join(save_dir, self.model_name + '_Er.pkl')):
            self.Er.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_Er.pkl')))
            
        if os.path.exists(os.path.join(save_dir, self.model_name + '_Ez.pkl')):
            self.Ez.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_Ez.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['Ez_loss']))

        y1 = hist['Er_loss']
        y2 = hist['Ez_loss']

        plt.plot(x, y1, label='Er_loss')
        plt.plot(x, y2, label='Ez_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)
