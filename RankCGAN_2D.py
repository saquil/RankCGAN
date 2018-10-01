import utils, torch, time, os, pickle, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from dataloader import dataloader
from dataloader import pairloader

class generator(nn.Module):
    # Network Architecture is exactly same as in DCGAN
    def __init__(self, input_dim=100, output_dim=1, input_size=64, rank_code=2):
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

class discriminator(nn.Module):
    # Network Architecture is exactly same as in DCGAN
    def __init__(self, input_dim=3, output_dim=1, input_size=64, rank_code=2):
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

class ranker(nn.Module):
    # Network Architecture is exactly same as the D in DCGAN  
    def __init__(self, input_dim=3, output_dim=2, input_size=64):
        super(ranker, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size        

        self.nconvs = int(math.log2(self.input_size)) - 3
        mult = 1
        layers = []

        
        layers.append(nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ))
        
        for i in range(0, self.nconvs):
            layers.append(nn.Sequential(
            nn.Conv2d(64 * mult, 64 * mult * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * mult * 2),
            nn.LeakyReLU(0.2),
            ))
            mult = mult * 2
      
        self.main = nn.Sequential(*layers)
        self.ranks = []
        for i in range(self.output_dim):
            self.ranks.append(nn.Sequential(
            nn.Conv2d(mult * 64, 1, 4),
            ))
        self.rs = nn.Sequential(*self.ranks)
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.main(input)
        outputs = []
        for i in range(self.output_dim):
            outputs.append(self.ranks[i](x).view([-1,1]))
        outputs = torch.cat(outputs,1)
        return outputs

class RankCGAN_2D(object):
    def __init__(self, args, SUPERVISED=True):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 100
        self.rank_code = 2         # rank latent variable
        self.sample_num = 100

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        self.spair_loader = pairloader(self.dataset, 'sporty', self.input_size, self.batch_size//2)
        self.bpair_loader = pairloader(self.dataset, 'black', self.input_size, self.batch_size//2)
        data = self.data_loader.__iter__().__next__()
        spair = self.spair_loader.__iter__().__next__()
        bpair = self.bpair_loader.__iter__().__next__()
        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, rank_code=self.rank_code)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, rank_code=self.rank_code)
        self.R = ranker(input_dim=data.shape[1], output_dim=self.rank_code, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.R_optimizer = optim.Adam(self.R.parameters(), lr=args.lrR, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.R.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        utils.print_network(self.R)
        print('-----------------------------------------------')

        # fixed noise & condition

        if self.rank_code == 1:
            lines = int(np.floor(np.sqrt(self.sample_num)))
            self.sample_z = torch.zeros((self.sample_num, self.z_dim))
            for i in range(lines):
                self.sample_z[i * lines] = torch.randn(1, self.z_dim)
                for j in range(1, lines):
                    self.sample_z[i * lines + j] = self.sample_z[i * lines]

            self.sample_y = torch.linspace(-1,1,lines).expand(lines, lines).contiguous().view([-1,1])
        elif self.rank_code ==2:
            lines = int(np.floor(np.sqrt(self.sample_num)))
            self.sample_z = torch.randn(1,self.z_dim).expand((self.sample_num, self.z_dim))
            self.sample_y = torch.cat([torch.linspace(-1,1,lines).expand(lines, lines).contiguous().view([-1,1]), torch.linspace(-1,1,lines).expand(lines, lines).t().contiguous().view([-1,1])],1)
            

        if self.gpu_mode:
            self.sample_z, self.sample_y = self.sample_z.cuda(), self.sample_y.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['R_loss'] = []
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size), torch.zeros(self.batch_size)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        self.R.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, x_ in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                x1_1, x2_1, r_1 = self.spair_loader.__iter__().__next__()
                x1_2, x2_2, r_2 = self.bpair_loader.__iter__().__next__()
		
                z_ = torch.randn((self.batch_size, self.z_dim))
                y_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size,self.rank_code))).type(torch.FloatTensor)
                

                if self.gpu_mode:
                    x_, z_, y_ = x_.cuda(), z_.cuda(), y_.cuda()
                    x1_1, x2_1, r_1 = x1_1.cuda(), x2_1.cuda(), r_1.float().cuda()
                    x1_2, x2_2, r_2 = x1_2.cuda(), x2_2.cuda(), r_2.float().cuda()
            
                x_p_1 = torch.cat([x1_1, x2_1],0)
                x_p_2 = torch.cat([x1_2, x2_2],0)

		        # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

		        #update R network
                self.R_optimizer.zero_grad()
                
                R_1 = self.R(x_p_1)[:,0]
                diff1 = R_1[:self.batch_size//2] - R_1[self.batch_size//2:]
                sig1 = nn.Sigmoid().cuda()
                prob1 = sig1(diff1)
                R1_loss = self.BCE_loss(prob1, r_1)
                
                R_2 = self.R(x_p_2)[:,1]
                diff2 = R_2[:self.batch_size//2] - R_2[self.batch_size//2:]
                sig2 = nn.Sigmoid().cuda()
                prob2 = sig2(diff2)
                R2_loss = self.BCE_loss(prob2, r_2)               

                R_loss = R1_loss + R2_loss
                	
                R_loss.backward()
                self.train_hist['R_loss'].append(R_loss.item())	
                self.R_optimizer.step()		

                # update G network
                self.G_optimizer.zero_grad()
                self.D_optimizer.zero_grad()

                G_ = self.G(z_, y_)
                D_fake = self.D(G_)

                G_D_loss = self.BCE_loss(D_fake, self.y_real_)

                R_fake = self.R(G_)
                diff_G = R_fake[:self.batch_size//2] - R_fake[self.batch_size//2:]
                diff_y = torch.ge(y_[:self.batch_size//2], y_[self.batch_size//2:]).float().squeeze()
                sig_G = nn.Sigmoid().cuda()
                prob_G = sig_G(diff_G)
                G_R_loss = self.BCE_loss(prob_G, diff_y)

                G_loss = G_D_loss + G_R_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()



                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, rank_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), R_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
        	    self.visualize_results((epoch+1))

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

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        y3 = hist['R_loss']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')
        plt.plot(x, y3, label='R_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)
