import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lagomorph as lm
from lagomorph import adjrep 
from lagomorph import deform 

class Epdiff():
    def __init__(self,alpha=2.0,gamma=1.0,img_size=64,steps=7):
        # alpha=2.0;gamma = 1.0
        fluid_params = [alpha, 0, gamma]; 
        self.metric = lm.FluidMetric(fluid_params)
        iden = deform.identity((1, 2, img_size, img_size))

        self.iden = torch.from_numpy(iden).cuda()
        self.num_steps = steps


    def EPDiff_step(self, m0, dt, phiinv, mommask=None):
        m = adjrep.Ad_star(phiinv, m0)
        if mommask is not None:
            m = m * mommask
        v = self.metric.sharp(m)
        return deform.compose_disp_vel(phiinv, v, dt=-dt), m, v
    

    def my_expmap(self, m0=None, v0=None, T=1.0, phiinv=None, mommask=None, checkpoints=False):
        num_steps = self.num_steps
        # t1 = default_timer()
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.
        What we return is actually only the inverse transformation phi^{-1}
        """
        m_seq=[]; v_seq=[]; u_seq=[]
        if m0 is None:
            d = len(v0.shape)-2
            m0 = self.metric.flat(v0)
        else:
            d = len(m0.shape)-2
            v0 = self.metric.sharp(m0)

        m_seq.append(m0); v_seq.append(v0)

        if phiinv is None:
            phiinv = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps): #num_steps=7  V0~V6
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                u_seq.append(phiinv)
                if i<(num_steps-1):
                    m_seq.append(m); v_seq.append(v)
        # print("my_expmap: {}".format(default_timer()-t1))
        return u_seq,v_seq,m_seq
    

    def my_expmap_u2phi(self, m0=None, v0=None, T=1.0, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        num_steps = self.num_steps
        m_seq=[]; v_seq=[]; u_seq=[];ui_seq=[]
        if m0 is None:
            d = len(v0.shape)-2
            m0 = self.metric.flat(v0)
        else:
            d = len(m0.shape)-2
            v0 = self.metric.sharp(m0)

        m_seq.append(m0); v_seq.append(v0)

        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                # print("~~~~m0~~~",torch.max(m0), torch.min(m0))
                # print("~~~~phiinv~~~",torch.max(phiinv), torch.min(phiinv))
                # print("~~~~m~~~",torch.max(m), torch.min(m))
                # print("~~~~v~~~",torch.max(v), torch.min(v))
                u_seq.append(phiinv)
                phi = phi + dt*lm.interp(v, phi)
                ui_seq.append(phi)

                if i<(num_steps-1):
                    m_seq.append(m); v_seq.append(v)
        
        phiinv_seq = [u+self.iden for u in u_seq]
        phi_seq = [ui+self.iden for ui in ui_seq]
        
        
        # Dispinv_List_gt, Disp_List_gt, V_List_gt
        return u_seq, ui_seq, v_seq, phiinv_seq, phi_seq, m_seq
    

    def my_expmap_shooting(self, m0, T=1.0, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        num_steps = self.num_steps
        # m_seq=[]; v_seq=[]; u_seq=[]
        d = len(m0.shape)-2

        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                phi = phi + dt*lm.interp(v, phi)
        
        return phiinv, phi

    def lagomorph_expmap_shootin(self, m0, T=1.0, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        num_steps = self.num_steps

        d = len(m0.shape)-2

        if phiinv is None:
            phiinv = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(self.metric, m0, dt, phiinv, mommask=mommask)
                
        return phiinv
    
    def my_get_u(self, v_seq=None, m_seq=None, T=1.0, phiinv=None):
        num_steps = self.num_steps
        if v_seq is None:
            if m_seq is None:
                assert 400>900
            v_seq = [self.metric.sharp(m) for m in m_seq]
        elif m_seq is None:
            if v_seq is None:
                assert 400>900
            m_seq = [self.metric.flat(v) for v in v_seq]
        
        dt = T/num_steps
        if phiinv is None:
            phiinv = torch.zeros_like(v_seq[0])

        u_seq = [];phiinv_seq=[]
        for i in range(num_steps): #num_steps=7  V0~V6 
            phiinv = deform.compose_disp_vel(phiinv, v_seq[i], dt=-dt)
            u_seq.append(phiinv)
            # print(torch.max(phiinv))
        # phiinv_seq = [u+self.iden for u in u_seq]
        return u_seq, m_seq, v_seq
        
    

    def my_get_u2phi(self, v_seq=None, m_seq=None, T=1.0, phiinv=None):
        num_steps = self.num_steps
        # t1 = default_timer()
        if v_seq is None:
            if m_seq is None:
                assert 400>900
            v_seq = [self.metric.sharp(m) for m in m_seq]
        elif m_seq is None:
            if v_seq is None:
                assert 400>900
            m_seq = [self.metric.flat(v) for v in v_seq]
        
        dt = T/num_steps
        if phiinv is None:
            phiinv = torch.zeros_like(v_seq[0])
            phi = torch.zeros_like(v_seq[0])

        u_seq = [];phiinv_seq=[];
        ui_seq = [];phi_seq=[]
        for i in range(num_steps):
            phiinv = deform.compose_disp_vel(phiinv, v_seq[i], dt=-dt)
            u_seq.append(phiinv)
            # print(torch.max(phiinv))
            # phiinv_seq = [u+self.iden(1, 2, 32,32) for u in u_seq]

            phi = phi + dt*lm.interp(v_seq[i], phi)
            ui_seq.append(phi)
            
        phiinv_seq = [u+self.iden for u in u_seq]
        phi_seq = [ui+self.iden for ui in ui_seq]

        return u_seq, ui_seq, phiinv_seq, phi_seq, m_seq


    def my_expmap_advect(self, m, T=1.0, phiinv=None):
        """Compute EPDiff with vector momenta without using the integrated form.

        This is Euler integration of the following ODE:
            d/dt m = - ad_v^* m
        """
        num_steps = self.num_steps
        v_seq = []; m_seq=[]
        d = len(m.shape)-2
        v0 = self.metric.sharp(m)
        m_seq.append(m); v_seq.append(v0)


        if phiinv is None:
            phiinv = torch.zeros_like(m)
        dt = T/num_steps
        v = self.metric.sharp(m)
        phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
        v_seq.append(v); m_seq.append(m)


        for i in range(num_steps-1):
            m = m - dt*adjrep.ad_star(v, m)
            v = self.metric.sharp(m)
            phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
            if i<(num_steps-2):
                v_seq.append(v); m_seq.append(m)
        return phiinv,v_seq,m_seq




class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode=None):
        if mode is None:
            mode = self.mode
        # new locations
        new_locs = self.grid + flow   #self.grid:  identity
        new_locs_unnormalize = self.grid + flow
        shape = flow.shape[2:]
        #  new_locs  :  torch.Size([1, 3, 64, 64, 64])
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 1) #[1, 64, 64, 64,3]
            new_locs_unnormalize = new_locs_unnormalize[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 4, 1)
            new_locs_unnormalize = new_locs_unnormalize[..., [2, 1, 0]]

        warped = F.grid_sample(src, new_locs, mode=mode)
        # print(new_locs.shape)   #[b, 64, 64, 64, 3]
        # print(warped.shape)     #[6, 3, 64, 64, 64]
        # return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

       
        return (warped, new_locs_unnormalize)



class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, registration=False):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)
        self.registration = registration
    def forward(self, vec):  ###速度场->形变场
        dispList = []

        vec = vec * self.scale
        dispList.append(vec)

        for _ in range(self.nsteps):
            scratch,_ = self.transformer(vec, vec)
            vec = vec + scratch
            dispList.append(vec)
        # print("vec ", vec.requires_grad)
        if not self.registration:
            return vec
        else:
            return vec, dispList





class Svf(nn.Module):
    def __init__(self, inshape, steps=7):
        super().__init__()
        self.nsteps = steps
        assert self.nsteps >= 0, 'nsteps should be >= 0, found: %d' % self.nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)
    

    # def integrate(self, pos_flow):
    # def Svf_shooting(self, pos_flow):  #pos_flow: [b, 2, 64, 64]  (b,64,64,2)
    def forward(self, pos_flow):  #pos_flow: [b, 2, 64, 64]  (b,64,64,2)
        dims = len(pos_flow.shape)-2
        if dims == 2:
            b,c,w,h = pos_flow.shape
            if c != 2 and c != 3:
                pos_flow = pos_flow.permute(0,3,1,2)
        elif dims == 3:
            b,c,w,h,d = pos_flow.shape
            if c != 3:
                pos_flow = pos_flow.permute(0,4,1,2,3)

        vec = pos_flow
        dispList = []
        
        vec = vec * self.scale
        # dispList.append(vec)


        for _ in range(self.nsteps):
            scratch,_ = self.transformer(vec, vec)
            vec = vec + scratch
            dispList.append(vec)

            # print(vec.shape)     #[70, 2, 64, 64]
            # assert 4>8888
        
        return vec, dispList   #len





class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        if(len(y_pred.shape) == 5):
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0

            if self.loss_mult is not None:
                grad *= self.loss_mult
            return grad
        elif(len(y_pred.shape) == 4):
            # print("y_pred   ",y_pred.shape)
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
          
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0

            if self.loss_mult is not None:
                grad *= self.loss_mult
            return grad








def SmoothOper(para=(2.0,1.0,2.0), iamgeSize=(128,128,128)):  #shape : [20, 64, 64, 64, 3]
    spx = 1 ##spacing information x 
    spy = 1 ##spacing information y 
    spz = 1 ##spacing information z 

   

    alpha = para[0]
    gamma = para[1]
    lpow = para[2]


    if(len(iamgeSize)==3):
        size_x, size_y, size_z = iamgeSize[0], iamgeSize[1], iamgeSize[2]
        gridx = torch.tensor(np.linspace(0, 1-1/size_x, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([1, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1-1/size_y, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([1, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1-1/size_z, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([1, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1)

        # trun1 = grid[:, :half_mode1, :half_mode1, :half_mode1]     #[b, modes, modes, modes, 3]
        # trun2 = grid[:, -half_mode2:, :half_mode1, :half_mode1]    #[b, modes, modes, modes, 3]
        # trun3 = grid[:, :half_mode1, -half_mode2:, :half_mode1]    #[b, modes, modes, modes, 3]
        # trun4 = grid[:, -half_mode2:, -half_mode2:, :half_mode1]   #[b, modes, modes, modes, 3]
        # yy1 = torch.cat((trun1,trun2),dim=-4)       #[b, 2*modes, modes, modes, 3]      #[b, 16, 8, 8, 3]
        # yy2 = torch.cat((trun3,trun4),dim=-4)       #[b, 2*modes, modes, modes, 3]      #[b, 16, 8, 8, 3]
        # trunr = torch.cat((yy1,yy2),dim=-3)         #[b, 2*modes, 2*modes, modes, 3]    #[b, 16, 16, 8, 3]

        coeff = (-2.0*torch.cos(2.0 * torch.pi * grid) + 2.0)/(spx*spx)
        val = pow(alpha*(torch.sum(coeff,dim=-1))+gamma, lpow)

    else:
        size_x, size_y = iamgeSize[0], iamgeSize[1]
        # gridx = torch.tensor(np.linspace(0, 1-1/size_x, size_x), dtype=torch.float)
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([1, 1, size_y, 1])

        # gridy = torch.tensor(np.linspace(0, 1-1/size_y, size_y), dtype=torch.float)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([1, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)


        # trun1 = grid[:, :half_mode1, :half_mode1]     #[b, modes, modes, 3]
        # trun2 = grid[:, -half_mode2:, :half_mode1]    #[b, modes, modes, 3]
        # trunr = torch.cat((trun1,trun2),dim=-3)       #[b, 2*modes, modes, 3]      #[b, 16, 8, 2]
        # trunr = grid

        ''' print(grid[0,:17, :17,0])
        print(grid[0,:17, :17,1])
        print(grid.shape)
        assert 2>222 '''
        coeff = (-2.0*torch.cos(2.0 * torch.pi * grid) + 2.0)/(spx*spx)
        val = pow(alpha*(torch.sum(coeff,dim=-1))+gamma, lpow)

        # Lcoeff = torch.stack((val,val),dim=-1)       #[b, 16, 8, 2]   sharp
        # Kcoeff = torch.stack((1/val,1/val),dim=-1)   #[b, 16, 8, 2]   smooth

    resSmooth = (1/val).squeeze(0)  #momemtum -> velocity
    # np.save("/home/nellie/code/cvpr/coshnet/nellie-grad/resSmooth.npy", resSmooth.numpy())
    # assert 3>124


    resSharp = val.squeeze(0)  #velocity -> momemtum

    
    # print(resSmooth[:5, :5])
    # print(resSmooth[-5:, :5])
    # print(resSmooth[:5, -5:])
    # print(resSmooth[-5:, -5:])


    # print(alpha, gamma)
    # assert 4>333

    
    return resSmooth, resSharp   ##[b, 16, 8]















