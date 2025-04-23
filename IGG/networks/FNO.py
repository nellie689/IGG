import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights): #input:[b, 20, 12, 12]   weights:[20, 20, 12, 12]
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # print(input.shape, weights.shape)
        '''1*'''
        # method1
        # result =  torch.einsum("bixy,ioxy->boxy", input, weights)  #[20, 20, 12, 12]

        '''2*'''
        ##  more effcient
        # x1 = torch.einsum("bixy,ioxy->boxy", input, weights[:,:4,...])  #[1, 20, 16, 16, 14]
        # x2 = torch.einsum("bixy,ioxy->boxy", input, weights[:,4:8,...])  #[1, 20, 16, 16, 14]
        # x3 = torch.einsum("bixy,ioxy->boxy", input, weights[:,8:12,...])  #[1, 20, 16, 16, 14]
        # x4 = torch.einsum("bixy,ioxy->boxy", input, weights[:,12:16,...])  #[1, 20, 16, 16, 14]
        # x5 = torch.einsum("bixy,ioxy->boxy", input, weights[:,16:,...])  #[1, 20, 16, 16, 14]
        # result = torch.concat((x1,x2,x3,x4,x5), dim=1)
    


        # '''3*''' bixy,ioxy->boxy  bixy*o1ixy->obixy
        weights_permuted = weights.permute(1,0,2,3).unsqueeze(1)
        # k = (input*(weights.permute(1,0,2,3).unsqueeze(1))).permute(1,0,2,3,4)
        k_raw = (input*weights_permuted)
        # input_real = input.real
        # weights_permuted_real = weights_permuted.real
        # k_raw_real = (input_real*weights_permuted_real)        
        k = k_raw.permute(1,0,2,3,4)
        result = torch.sum(k,dim=2)   #[10, 20, 8, 8]
        # # print(torch.allclose(result,result1))



        return result
    


    def forward(self, x):   #Size([25, 20, 2, 2])
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x) #[1, 20, 128, 65]
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)  #[20, 20, 64, 33]
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)        
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) #out_ft:[20, 20, 128, 65]    x:[20, 20, 128, 128]
        return x



class FNO2d(nn.Module):
    def __init__(self, modes1=2, modes2=2, width=20, Config=None, SME16=None):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(20, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 20)

        self.Config = Config
        if "SMFNO" in self.Config["general"]["sub_version"]:
            self.register_buffer('SME16', SME16/4)

    def forward(self, x):  #([25, 32, 2, 2]
        x = x.permute(0,2,3,1)   #([25, 32, 2, 2]   ->[25, 2, 2, 32]

        # grid = self.get_grid(x.shape, x.device) #[20, 64, 64, 2]
        # x = torch.cat((x, grid), dim=-1)        #[20, 64, 64, 3]
        x = self.fc0(x)                         #[20, 64, 64, 20]   Linear(in_features=12, out_features=20, bias=True)
        x = x.permute(0, 3, 1, 2)               #Size([25, 32, 2, 2])
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        # t1 = default_timer()
        x1 = self.conv0(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        # t2 = default_timer()
        x2 = self.w0(x)                         #[20, 20, 64, 64]   Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1))
        # t3 = default_timer()
        # print(f"!!!!!!!!!!! conv0: {t2-t1}, w0: {t3-t2}")
        x = x1 + x2
        x = F.gelu(x)

        if self.Config is not None and "SMFNO" in self.Config["general"]["sub_version"]:
            x = self.SmoothSignal(x, size=16)


        x1 = self.conv1(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        if self.Config is not None and "SMFNO" in self.Config["general"]["sub_version"]:
            x = self.SmoothSignal(x, size=16)

        x1 = self.conv2(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        if self.Config is not None and "SMFNO" in self.Config["general"]["sub_version"]:
            x = self.SmoothSignal(x, size=16)

        x1 = self.conv3(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        x2 = self.w3(x)
        x = x1 + x2
        if self.Config is not None and "SMFNO" in self.Config["general"]["sub_version"]:
            x = self.SmoothSignal(x, size=16)
        
        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)               #[20, 64, 64, 20]
        x = self.fc1(x)                         #[20, 64, 64, 128]          Linear(in_features=20, out_features=128, bias=True)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        if self.Config is not None and "SMFNO" in self.Config["general"]["sub_version"]:
            x = self.SmoothSignal(x, size=16)                         #[20, 64, 64, 2]            Linear(in_features=128, out_features=1, bias=True)
        return x

    def get_grid(self, shape, device):    #shape:[20, 64, 64, 10]
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  #[64]
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  #[20, 64, 64, 1]
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def SmoothSignal(self, x, size=16):
        # print(x.shape, self.SME16.shape)
        # assert 2>333
        x = torch.fft.rfftn(x, dim=(-2, -1))
        # print(x.shape, self.SME16.shape)
        if size == 16:
            x = x * self.SME16
        return torch.fft.irfftn(x, dim=(-2, -1))
