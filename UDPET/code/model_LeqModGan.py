from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from nets_GAN import Unet, gaussian_weights_init, Discriminator
from torch.optim import lr_scheduler
from torch.nn import MSELoss, L1Loss


class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real, phase=None):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return (input - target_tensor).pow(2).mean()


class modelGAN(nn.Module):
    def __init__(self, opts):
        super(modelGAN, self).__init__()
        self.optimizers = []
        self.loss_names = []
        self.is_train = True if hasattr(opts, 'lr') else False
        self.opts = opts

        self.net_G = Unet(inshape=opts.patch_size, nb_features=[[48, 96, 192, 384], [384, 192, 96, 48, 24, 1]])
        num_param = sum([p.numel() for p in self.net_G.parameters() if p.requires_grad])
        print('Number of generator parameters: {}'.format(num_param))
        
        self.net_D = Discriminator(in_channels=1, nf=32, norm_layer='IN')
        num_param = sum([p.numel() for p in self.net_D.parameters() if p.requires_grad])
        print('Number of discriminator parameters: {}'.format(num_param))
        
        self.net_G.to(opts.device)
        self.net_D.to(opts.device)
        if opts.device.type == 'cuda' and opts.numGPUs >= 1:
            self.net_G = nn.DataParallel(self.net_G, device_ids=range(opts.numGPUs))
            self.net_D = nn.DataParallel(self.net_D, device_ids=range(opts.numGPUs))

        if self.is_train:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr, betas=(0.9, 0.999), weight_decay=1e-7)
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opts.lr, betas=(0.9, 0.999), weight_decay=1e-7)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            self.loss_names += ['loss_D', 'loss_G_GAN', 'loss_recon']
            if opts.weightLoss_localSUVbias is not None:
                self.loss_names += ['loss_localSUVbias']
            if opts.weightLoss_lesionSUVbias is not None:
                self.loss_names += ['loss_lesionSUVbias']

    @staticmethod
    def _unwrap(module):
        return module.module if isinstance(module, nn.DataParallel) else module

    def initialize(self):
        if self.opts.preWeight is not None:
            checkpoint = torch.load(self.opts.preWeight, map_location=self.opts.device)
            self._unwrap(self.net_G).load_state_dict(checkpoint['net_G'], strict=True)
            self._unwrap(self.net_D).load_state_dict(checkpoint['net_D'], strict=True)
            print('generator and discriminator intialized from: ', self.opts.preWeight)
        else:
            self.net_G.apply(gaussian_weights_init)
            self.net_D.apply(gaussian_weights_init)
            print('generator and discriminator intialized from Gaussian_weights')

    def set_scheduler(self, opts, epoch=-1):
        if opts.lr_policy == 'multistep':
            self.schedulers = [lr_scheduler.MultiStepLR(optimizer, milestones=opts.multi_step_size, gamma=opts.gamma, last_epoch=-1, verbose=True) for optimizer in self.optimizers]
        elif opts.lr_policy == 'cosine':
            self.schedulers = [lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.n_epochs, eta_min=0, last_epoch=-1, verbose=True) for optimizer in self.optimizers]
        elif opts.lr_policy == 'ReduceLROnPlateau':
            self.schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opts.gamma, patience=opts.Plateau_step_size, cooldown=1, verbose=True) for optimizer in self.optimizers]
        else:
            NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)

    def set_input(self, data):
        self.inp_vol_low = data['vol_low'].flatten(0,1).to(self.opts.device).float().unsqueeze(dim=1)  # [B,1,X,Y,Z]
        self.inp_vol_high = data['vol_high'].flatten(0,1).to(self.opts.device).float().unsqueeze(dim=1)
        if self.opts.weightLoss_lesionSUVbias is not None:
            self.vol_weight = data['vol_weight'].flatten(0,1).to(self.opts.device).float().unsqueeze(dim=1)  # [B,1,X,Y,Z]

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def optimize(self, total_iter):
        # ttttttt = torch.argwhere(self.vol_weight[0,0,:,:,:]==1)
        # ttttttt = ttttttt[len(ttttttt)//2][0] if len(ttttttt)>=1 else self.inp_vol_low.shape[2]//2
        # plt.imsave('c1Fig.png', self.inp_vol_low[0,0,ttttttt,:,:].detach().cpu().numpy(), cmap='jet', vmin=0.03, vmax=5.0)
        # plt.imsave('c2Fig.png', self.inp_vol_high[0,0,ttttttt,:,:].detach().cpu().numpy(), cmap='jet', vmin=0.03, vmax=5.0)
        # plt.imsave('c3Fig.png', self.vol_weight[0,0,ttttttt,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1.0)
        # ========================================D============================================
        self.net_D.zero_grad()
        # fake
        self.pred_vol_high = self.net_G(self.inp_vol_low)
        pred_fake = self.net_D(self.pred_vol_high.detach())
        loss_D_fake = LSGANLoss().to(pred_fake.device)(pred_fake, target_is_real=False)
        # real
        pred_real = self.net_D(self.inp_vol_high.detach())
        loss_D_real = LSGANLoss().to(pred_real.device)(pred_real, target_is_real=True)
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        
        if total_iter % 1 == 0:
            self.loss_D.backward()
            self.optimizer_D.step()
        # =======================================G=============================================
        self.net_G.zero_grad()
        self.pred_vol_high = self.net_G(self.inp_vol_low)
        pred_fake = self.net_D(self.pred_vol_high)
        self.loss_G_GAN = LSGANLoss().to(pred_fake.device)(pred_fake, target_is_real=True)
        
        self.loss_recon = self.opts.weightLoss_mse * MSELoss(reduction='mean')(self.pred_vol_high, self.inp_vol_high)
        self.lossAll = self.loss_recon.clone()

        if self.opts.weightLoss_localSUVbias is not None: # multiscale quantification-consistency constraint
            self.loss_localSUVbias = 0.0
            for (ppsSize, ppsStep, ppsCoe) in zip((5,9,17,33), (2,4,8,16), (0.8, 0.15, 0.03, 0.02)):
                pred_pps = self.pred_vol_high.clone().unfold(2,ppsSize,ppsStep).unfold(3,ppsSize,ppsStep). \
                                                      unfold(4,ppsSize,ppsStep).contiguous().view(-1,ppsSize,ppsSize,ppsSize)
                gt_pps = self.inp_vol_high.clone().unfold(2,ppsSize,ppsStep).unfold(3,ppsSize,ppsStep). \
                                                   unfold(4,ppsSize,ppsStep).contiguous().view(-1,ppsSize,ppsSize,ppsSize)
                self.loss_localSUVbias += ppsCoe * L1Loss(reduction='mean')(nn.AdaptiveAvgPool3d((1,1,1))(pred_pps.unsqueeze(dim=1)), 
                                                                            nn.AdaptiveAvgPool3d((1,1,1))(gt_pps.unsqueeze(dim=1))) \
                                        + ppsCoe * L1Loss(reduction='mean')(nn.AdaptiveMaxPool3d((1,1,1))(pred_pps.unsqueeze(dim=1)), 
                                                                            nn.AdaptiveMaxPool3d((1,1,1))(gt_pps.unsqueeze(dim=1)))
            self.loss_localSUVbias = self.opts.weightLoss_localSUVbias * self.loss_localSUVbias
            self.lossAll += self.loss_localSUVbias
        
        if self.opts.weightLoss_lesionSUVbias is not None: # lesion-perceived consistency constrint
            segValidIndex = [self.vol_weight >= self.opts.segProbThresh]
            if self.pred_vol_high[segValidIndex].shape[0] > 0:
                loss_totalTumorBias = L1Loss(reduction='none')(self.pred_vol_high[segValidIndex], self.inp_vol_high[segValidIndex]) * self.vol_weight[segValidIndex]
                loss_totalTumorBias = loss_totalTumorBias.mean()
                self.loss_lesionSUVbias = self.opts.weightLoss_lesionSUVbias * loss_totalTumorBias
                self.lossAll += self.loss_lesionSUVbias
            else:
                self.loss_lesionSUVbias = torch.zeros_like(self.loss_recon)

        (self.loss_G_GAN + self.lossAll).backward()
        self.optimizer_G.step()

    # @property
    def loss_summary(self, ):
        message = 'loss_D: {:4f}, loss_G_GAN: {:4f}, loss_recon: {:4f}'.format(self.loss_D.item(), self.loss_G_GAN.item(), self.loss_recon.item())
        if self.opts.weightLoss_localSUVbias is not None:
            message += ' loss_localSUVbias: {:4f}'.format(self.loss_localSUVbias.item())
        if self.opts.weightLoss_lesionSUVbias is not None:
            message += ' loss_lesionSUVbias: {:4f}'.format(self.loss_lesionSUVbias.item())
        return message

    def update_learning_rate(self, epochLossAll):
        for scheduler in self.schedulers:
            if (self.opts.lr_policy == 'multistep') or (self.opts.lr_policy == 'cosine'):
                scheduler.step()
            elif self.opts.lr_policy == 'ReduceLROnPlateau':
                scheduler.step(epochLossAll)
        
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))
        return lr

    def save(self, filename, epoch, total_iter):
        state = {}
        state['net_G'] = self._unwrap(self.net_G).state_dict()
        state['net_D'] = self._unwrap(self.net_D).state_dict()
        state['opt_G'] = self.optimizer_G.state_dict()
        state['opt_D'] = self.optimizer_D.state_dict()
        state['epoch'] = epoch
        state['total_iter'] = total_iter
        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.opts.device)
        self._unwrap(self.net_G).load_state_dict(checkpoint['net_G'])
        self._unwrap(self.net_D).load_state_dict(checkpoint['net_D'])
        if train:
            self.optimizer_G.load_state_dict(checkpoint['opt_G'])
            self.optimizer_D.load_state_dict(checkpoint['opt_D'])

        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']
