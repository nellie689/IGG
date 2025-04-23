import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Util.EpdiffLib import Epdiff, Svf, SmoothOper
from self_attention_cv import ViT
import lagomorph as lm
from .FNO import FNO2d


from tensorboardX import SummaryWriter



class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernal=3, stride=1, padding=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, Config=None):
        super(Encoder, self).__init__()
        self.Config = Config

        self.gradlist = []
        self.graph_flag = False

        img_size = Config['DataSet']['train_img_size']
        inshape=(img_size, img_size)
        
        TSteps = Config['general']['num_steps']
        self.MSvf = Svf(inshape=(img_size, img_size), steps=TSteps)
        self.img_size = img_size
        if Config["general"]["level"] == 1:
            # nb_features=[[32, 64, 20], [20, 32, 32, 32, 16, 16]]
            nb_features=[[16, 32, 20], [20, 32, 32, 32, 16, 16]]
        elif Config["general"]["level"] == 0:
            nb_features=[[10, 20, 20], [20, 20, 20, 20, 10, 10]]
       

        nb_features=nb_features

        
        if self.Config["general"]["sub_version"].startswith("F"):
            infeats=4
        elif self.Config["general"]["sub_version"].startswith("S"):
            infeats=2
            if Config["DataSet"]["dataName"] in ["Plant"]:
                infeats=6
        elif self.Config["general"]["sub_version"].startswith("Cosh"):
            infeats=40
        else:
            infeats=2

        nb_levels=None
        max_pool=2
        feat_mult=1
        nb_conv_per_level=1
        half_res=False
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]


        self.encoder = nn.ModuleList()
        
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)

class MLP(nn.Module):
    def __init__(self, Config=None):
        super(MLP, self).__init__()
        self.Config = Config
        if self.Config["general"]["sub_version"] in {"F002", "F102"}:
            self.fc1 = nn.Linear(20 * 16 * 8, 1000)
            self.fc2 = nn.Linear(1000, 20*16*8)
        elif self.Config["general"]["sub_version"] in {"F003"}:
            self.fc1 = nn.Linear(20 * 16 * 8, 1000)
            self.fc2 = nn.Linear(1000, 4*16*8)
        elif self.Config["general"]["sub_version"] in {"S002", "S102", "CoshS102", "S102_SYMNet16_SVF_ABI"}:
            self.fc1 = nn.Linear(20 * 16 * 16, 1000)
            self.fc2 = nn.Linear(1000, 20*16*8)
        elif self.Config["general"]["sub_version"] in {"S003"}:
            self.fc1 = nn.Linear(20 * 16 * 16, 1000)
            self.fc2 = nn.Linear(1000, 4*16*8)
        elif self.Config["general"]["sub_version"] in {"S102_SYMNet16_SVF_FFT_ABI", "S102_SYMNet16_FFT_ABI", "S102_SYMNet16_UPS_ABI", "S102_SYMNet16_UPS2_ABI"}:
            self.fc1 = nn.Linear(20 * 16 * 16, 1000)
            self.fc2 = nn.Linear(1000, 20*16*16)




        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

class RemainConv(nn.Module):
    def __init__(self, Config=None):
        super(RemainConv, self).__init__()
        self.Config = Config

        self.gradlist = []
        self.graph_flag = False

        img_size = Config['DataSet']['train_img_size']
        inshape=(img_size, img_size)

        ndims = len(inshape)

        final_convs = [20, 10, 10]
        if self.Config["general"]["sub_version"] in {"S102_SYMNet16_SVF_FFT_ABI", "S102_SYMNet16_FFT_ABI", "S102_SYMNet16_UPS_ABI", "S102_SYMNet16_UPS2_ABI"}:
            prev_nf = 20
        elif "TT" in self.Config["general"]["sub_version"]:
            prev_nf = 20
        else:
            prev_nf = 10
        
       
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf


        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.final_nf, ndims, kernel_size=3, padding=1)

    
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm






class NetJuly(nn.Module):
    def __init__(self, Config=None):
        
        # super().__init__()
        super(NetJuly, self).__init__()
        self.Config = Config
        self.encoder = Encoder(Config)
        if "AETT" in self.Config["general"]["sub_version"]:
            print("11111111111111111111111111111\n\n")
            # self.SelfAttention1 = ViT(img_dim=64, in_channels=10, patch_dim=1, heads=1, num_classes=10, dim=10, \
            #         classification=False, blocks=Config["general"]["blocks"], dim_linear_block=Config["general"]["dim_linear_block"])
            self.SelfAttention2 = ViT(img_dim=32, in_channels=20, patch_dim=1, heads=1, num_classes=10, dim=20, \
                    classification=False, blocks=Config["general"]["blocks"], dim_linear_block=Config["general"]["dim_linear_block"])
            self.SelfAttention3 = ViT(img_dim=16, in_channels=20, patch_dim=1, heads=1, num_classes=10, dim=20, \
                    classification=False, blocks=Config["general"]["blocks"], dim_linear_block=Config["general"]["dim_linear_block"])

        elif "ATT" in self.Config["general"]["sub_version"] and not "SATT101" in self.Config["general"]["sub_version"]:
            print("22111111111111111111111111111\n\n")
            self.SelfAttention = ViT(img_dim=16, in_channels=20, patch_dim=1, heads=1, num_classes=10, dim=20, \
                    classification=False, blocks=Config["general"]["blocks"], dim_linear_block=Config["general"]["dim_linear_block"])
        
        if not "AETT" in self.Config["general"]["sub_version"] and not "ATT" in self.Config["general"]["sub_version"]: 
            print("33111111111111111111111111111\n\n")
            self.mlp = MLP(Config)
        

        self.remain = RemainConv(Config)

        
        

        img_size = Config['DataSet']['train_img_size']
        # img_size = 64
        TSteps = Config['general']['num_steps']
        self.TSteps = TSteps
        self.MSvf = Svf(inshape=(img_size, img_size), steps=TSteps)
        self.img_size = img_size

        crop_size = Config['DataSet']['crop_size'] #33
        self.half_mode1 = (crop_size // 2)+1
        self.half_mode2 = (crop_size // 2)
        
        #gamma=0.5  alpha=1.0   power=2  #GDN
        #gamma=1.0  alpha=2.0   power=3  #Smooth
        alpha = Config['LossWeightLddmm']['alpha']
        gamma = Config['LossWeightLddmm']['gamma']
        power = Config['LossWeightLddmm']['power']

        print("alpha", alpha, "gamma", gamma, "power", power)
        self.fluid_params = [alpha, 0, gamma]
        self.metric = lm.FluidMetric(self.fluid_params)
        self.MEpdiff = Epdiff(alpha=alpha, gamma=gamma, steps=TSteps, img_size=self.img_size)

        
        identity = torch.from_numpy(lm.identity((1,2,self.img_size,self.img_size))).cuda()
        self.register_buffer('id', identity)
        

        # gamma=1.0; alpha=4.0;power=2
        SmoothOperator, SharpOperator = SmoothOper(para=(alpha,gamma,power), iamgeSize=(img_size,img_size))
        SharpOperatorCenter = torch.fft.fftshift(SharpOperator, dim=(-2, -1))
        # np.save(f"/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/SmoothOperator/SmoothOperator_alpha{alpha}_gamma{gamma}_power{power}.npy", SmoothOperator.cpu().detach().numpy())
        # np.save(f"/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/SmoothOperator/SharpOperator_alpha{alpha}_gamma{gamma}_power{power}.npy", SharpOperator.cpu().detach().numpy())
        # print(alpha, gamma, power)
        # print("SmoothOperator", SmoothOperator.shape)
        # assert 2>333
        
        
        SmoothOperatorCenter = torch.fft.fftshift(SmoothOperator, dim=(-2, -1))
        SmoothOperatorCenter16 = SmoothOperatorCenter[64-8:64+8, 64-8:64+8]
        self.register_buffer('SmoothOperatorCenter16', SmoothOperatorCenter16)

        SMC16 = torch.fft.fftshift(SmoothOperator, dim=(-2, -1));SMC16 = SMC16[64-8:64+8, 64-8:64+8];self.register_buffer('SMC16', SMC16)
        SMC32 = torch.fft.fftshift(SmoothOperator, dim=(-2, -1));SMC32 = SMC32[64-16:64+16, 64-16:64+16];self.register_buffer('SMC32', SMC32)
        SMC64 = torch.fft.fftshift(SmoothOperator, dim=(-2, -1));SMC64 = SMC64[64-32:64+32, 64-32:64+32];self.register_buffer('SMC64', SMC64)
        SMC128 = torch.fft.fftshift(SmoothOperator, dim=(-2, -1));SMC128 = SMC128[64-64:64+64, 64-64:64+64];self.register_buffer('SMC128', SMC128)
        

        SMEH16 = torch.fft.fftshift(SmoothOperator, dim=(-2, -1));SMEH16 = SMEH16[64-8:64+8, 64-8:64+9]; SMEH16 = torch.fft.ifftshift(SMEH16, dim=(-2, -1));SMEH16 = SMEH16[:, :9];self.register_buffer('SMEH16', SMEH16) 
        SME16 = torch.zeros(16,9, dtype=SmoothOperator.dtype);SME16[:8, :] =  SmoothOperator[:8, :9]; SME16[-8:, :] = SmoothOperator[-8:, :9];self.register_buffer('SME16', SME16); 
        

        SMEH32 = torch.fft.fftshift(SmoothOperator, dim=(-2, -1));SMEH32 = SMEH32[64-16:64+16, 64-16:64+17]; SMEH32 = torch.fft.ifftshift(SMEH32, dim=(-2, -1));SMEH32 = SMEH32[:, :17];self.register_buffer('SMEH32', SMEH32)
        SME32 = torch.zeros(32,17, dtype=SmoothOperator.dtype);SME32[:16, :] =  SmoothOperator[:16, :17]; SME32[-16:, :] = SmoothOperator[-16:, :17];self.register_buffer('SME32', SME32);

        SMEH64 = torch.fft.fftshift(SmoothOperator, dim=(-2, -1));SMEH64 = SMEH64[64-32:64+32, 64-32:64+33]; SMEH64 = torch.fft.ifftshift(SMEH64, dim=(-2, -1));SMEH64 = SMEH64[:, :33];self.register_buffer('SMEH64', SMEH64)
        SME64 = torch.zeros(64,33, dtype=SmoothOperator.dtype);SME64[:32, :] =  SmoothOperator[:32, :33]; SME64[-32:, :] = SmoothOperator[-32:, :33];self.register_buffer('SME64', SME64);
        
        SMEH128 = SmoothOperator;SMEH128 = SMEH128[:, :65];self.register_buffer('SMEH128', SMEH128)
        SME128 = torch.zeros(128,65, dtype=SmoothOperator.dtype);SME128[:64, :] =  SmoothOperator[:64, :65]; SME128[-64:, :] = SmoothOperator[-64:, :65];self.register_buffer('SME128', SME128);


        SmoothOperator = SmoothOperator.type(torch.cuda.FloatTensor); SharpOperator = SharpOperator.type(torch.cuda.FloatTensor)
        SmoothOperator = SmoothOperator[:, :65]; SharpOperator = SharpOperator[:, :65]
        self.register_buffer('SmoothOperator', SmoothOperator); self.register_buffer('SharpOperator', SharpOperator)


        CropSmoothOperator11 = SmoothOperator[:33, :33]; CropSmoothOperator22 = SmoothOperator[-32:, :33]; 
        CropHfSmoothOperator = torch.zeros(65,33, dtype=SmoothOperator.dtype)
        CropHfSmoothOperator[:33, :] = CropSmoothOperator11; CropHfSmoothOperator[-32:, :] = CropSmoothOperator22
        CropSharpOperator11 = SharpOperator[:33, :33]; CropSharpOperator22 = SharpOperator[-32:, :33];
        CropHfSharpOperator = torch.zeros(65,33, dtype=SharpOperator.dtype)
        CropHfSharpOperator[:33, :] = CropSharpOperator11; CropHfSharpOperator[-32:, :] = CropSharpOperator22
        self.register_buffer('CropHfSmoothOperator', CropHfSmoothOperator); self.register_buffer('CropHfSharpOperator', CropHfSharpOperator)


        
        
        CropSmoothOperator1 = SmoothOperator[:17, :17]; CropSmoothOperator2 = SmoothOperator[-16:, :17]; 
        CropSmoothOperator = torch.zeros(33,17, dtype=SmoothOperator.dtype)
        CropSmoothOperator[:17, :] = CropSmoothOperator1; CropSmoothOperator[-16:, :] = CropSmoothOperator2
        CropSharpOperator1 = SharpOperator[:17, :17]; CropSharpOperator2 = SharpOperator[-16:, :17];
        CropSharpOperator = torch.zeros(33,17, dtype=SharpOperator.dtype)
        CropSharpOperator[:17, :] = CropSharpOperator1; CropSharpOperator[-16:, :] = CropSharpOperator2
        self.register_buffer('CropSmoothOperator', CropSmoothOperator); self.register_buffer('CropSharpOperator', CropSharpOperator)


        
        CropSmoothOperator111 = SmoothOperator[:9, :9]; CropSmoothOperator222 = SmoothOperator[-8:, :9]; 
        CropMinSmoothOperator = torch.zeros(17,9, dtype=SmoothOperator.dtype)
        CropMinSmoothOperator[:9, :] = CropSmoothOperator111; CropMinSmoothOperator[-8:, :] = CropSmoothOperator222
        CropSharpOperator111 = SharpOperator[:9, :9]; CropSharpOperator222 = SharpOperator[-8:, :9];
        CropMinSharpOperator = torch.zeros(17,9, dtype=SharpOperator.dtype)
        CropMinSharpOperator[:9, :] = CropSharpOperator111; CropMinSharpOperator[-8:, :] = CropSharpOperator222
        self.register_buffer('CropMinSmoothOperator', CropMinSmoothOperator); self.register_buffer('CropMinSharpOperator', CropMinSharpOperator)


        if Config["general"]["module_name"] in {"JulyGDN", "JulyGDN_DifuS"}:
            ModesFno=4; WidthFno=20
            ModesFno = Config["general"]["ModesFno"]
            WidthFno = Config["general"]["WidthFno"]

            self.model_v = FNO2d(modes1=ModesFno, modes2=ModesFno, width=WidthFno, Config=Config, SME16=self.SME16)
        if Config["general"]["SepFNODeCoder"] == "Yes":
            self.remain_model_v = RemainConv(Config)



    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
    
    # 冻结 self.encoder 和 self.SelfAttention
    def freeze_encoder_and_attention(self):
        if self.Config["general"]["SepFNODeCoder"] == "Yes":
            if hasattr(self, "remain"):
                for param in self.remain.parameters():
                    param.requires_grad = False
        if hasattr(self, "encoder"):
            for param in self.encoder.parameters():
                param.requires_grad = False
        if hasattr(self, "SelfAttention"):
            for param in self.SelfAttention.parameters():
                param.requires_grad = False
    # 解冻 self.model_v
    def unfreeze_model_v(self):
        if hasattr(self, "model_v"):
            for param in self.model_v.parameters():
                param.requires_grad = True
        if self.Config["general"]["SepFNODeCoder"] == "Yes":
            if hasattr(self, "remain_model_v"):
                for param in self.remain_model_v.parameters():
                    param.requires_grad = True
        else:
            if hasattr(self, "remain"):
                for param in self.remain.parameters():
                    param.requires_grad = True


    #解冻 self.encoder 和 self.SelfAttention
    def unfreeze_encoder_and_attention(self):
        if hasattr(self, "remain"):
            for param in self.remain.parameters():
                param.requires_grad = True
        if hasattr(self, "encoder"):
            for param in self.encoder.parameters():
                param.requires_grad = True
        if hasattr(self, "SelfAttention"):
            for param in self.SelfAttention.parameters():
                param.requires_grad = True
    # 冻结 self.model_v
    def freeze_model_v(self):
        if hasattr(self, "model_v"):
            for param in self.model_v.parameters():
                param.requires_grad = False
        if self.Config["general"]["SepFNODeCoder"] == "Yes":
            if hasattr(self, "remain_model_v"):
                for param in self.remain_model_v.parameters():
                    param.requires_grad = False
        










    def SmoothSignal(self, x, size=128):
        x = torch.fft.rfftn(x, dim=(-2, -1))
        if size == 128:
            x = x * self.SMEH128
        elif size == 64:
            x = x * self.SME64
        elif size == 32:
            x = x * self.SME32
        elif size == 16:
            x = x * self.SME16
        return torch.fft.irfftn(x, dim=(-2, -1))


    def deformSrc_SVF(self, v):
        u, u_seq = self.MSvf(v) #u:[42, 2, 128, 128]
        Sdef,phiinv = self.MSvf.transformer(self.src, u)   #sdef: [32, 1, 64, 64]  
        return Sdef, phiinv
    
    def defromSrcList_SVF(self, v):
        u, u_seq = self.MSvf(v)
        u_phi, u_seq_phi = self.MSvf(-v)
        SdefList = []; phiinvList = []; phiList = []
        for i in range(len(u_seq)):
            Sdef,phiinv = self.MSvf.transformer(self.src, u_seq[i])
            SdefList.append(Sdef)
            phiinvList.append(phiinv)
            _, phi = self.MSvf.transformer(self.src, u_seq_phi[i])
            phiList.append(phi)
        return SdefList, phiinvList, phiList
    
    def deformSrc_GDN(self, V_List):
        # V_List: V0~V6  Compare:V1~V6 
        #u_seq, m_seq, v_seq
        Dispinv_List_pred, M_List, _ = self.MEpdiff.my_get_u(v_seq=V_List)
        # print(self.src.shape,Dispinv_List_pred[-1].shape) #[60, 3, 128, 128]  [60, 2, 128, 128]
        Sdef = lm.interp(self.src, Dispinv_List_pred[-1])
        # u_seq, ui_seq, v_seq, phiinv_seq, phi_seq
        Dispinv_List_gt, Disp_List_gt, V_List_gt, Phiinv_List_gt, Phi_List_gt, M_list_gt = self.MEpdiff.my_expmap_u2phi(v0=V_List[0])
        return Sdef, V_List_gt, M_List
    
    def deformSrc_GDN_byGT(self, V0):
        Dispinv_List_gt, Disp_List_gt, V_List_gt, Phiinv_List_gt, Phi_List_gt, M_list_gt = self.MEpdiff.my_expmap_u2phi(v0=V0)
        Sdef = lm.interp(self.src, Dispinv_List_gt[-1])
        return Sdef, V_List_gt, M_list_gt
        
    def deformSrcList_GDN_byGT(self, V0):
        Dispinv_List_gt, Disp_List_gt, V_List_gt, Phiinv_List_gt, Phi_List_gt, M_list_gt = self.MEpdiff.my_expmap_u2phi(v0=V0)
        Sdef_List_gt = [lm.interp(self.src, Disp) for Disp in Dispinv_List_gt]


        return Sdef_List_gt, Phiinv_List_gt, Phi_List_gt, V_List_gt, M_list_gt, Dispinv_List_gt
        Sdef, V_List_gt, M_list_gt = self.deformSrc_GDN_byGT(V0) 
        Sdef_List, Phiinv_List, Phi_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt = self.deformSrcList_GDN(V_List)
    

    def deformSrcList_GDN(self, V_List, srcRGB=None):
        # print(self.src.shape, srcRGB.shape)
        # assert 2>111
        # V_List: V0~V6  Compare:V1~V6
        #u_seq, ui_seq, phiinv_seq, phi_seq
        Dispinv_List_pred, Disp_List_pred, Phiinv_List_pred, Phi_List_pred, M_List = self.MEpdiff.my_get_u2phi(v_seq=V_List)
        Sdef_List = [lm.interp(self.src, Disp) for Disp in Dispinv_List_pred]
        
        
        Dispinv_List_gt, Disp_List_gt, V_List_gt, Phiinv_List_gt, Phi_List_gt, M_list_gt = self.MEpdiff.my_expmap_u2phi(v0=V_List[0])
        Sdef_List_gt = [lm.interp(self.src, Disp) for Disp in Dispinv_List_gt]

        # SdefList, phiinvList, phiList, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List
        if srcRGB is not None:
            srcB = srcRGB[...,0]
            srcG = srcRGB[...,1]
            srcR = srcRGB[...,2]

            srcB_List = [lm.interp(srcB, Disp) for Disp in Dispinv_List_pred]
            srcG_List = [lm.interp(srcG, Disp) for Disp in Dispinv_List_pred]
            srcR_List = [lm.interp(srcR, Disp) for Disp in Dispinv_List_pred]
            srcRGB_List = [torch.stack([srcB, srcG, srcR], dim=-1) for srcB, srcG, srcR in zip(srcB_List, srcG_List, srcR_List)]

            srcB_List = [lm.interp(srcB, Disp) for Disp in Dispinv_List_gt]
            srcG_List = [lm.interp(srcG, Disp) for Disp in Dispinv_List_gt]
            srcR_List = [lm.interp(srcR, Disp) for Disp in Dispinv_List_gt]
            srcRGB_List_gt = [torch.stack([srcB, srcG, srcR], dim=-1) for srcB, srcG, srcR in zip(srcB_List, srcG_List, srcR_List)]
            # assert 4>901
            return Sdef_List, Phiinv_List_pred, Phi_List_pred, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List_pred, Phi_List_gt, M_list_gt, srcRGB_List, srcRGB_List_gt
        
        return Sdef_List, Phiinv_List_pred, Phi_List_pred, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List_pred, Phi_List_gt, M_list_gt
    
    def deformSrc_LDDMM(self, v0):
        # u_seq, ui_seq, v_seq, phiinv_seq, phi_seq
        Dispinv_List_gt, Disp_List_gt, V_List_gt, Phiinv_List_gt, Phi_List_gt, M_list_gt = self.MEpdiff.my_expmap_u2phi(v0=v0)
        Sdef = lm.interp(self.src, Dispinv_List_gt[-1])
        return Sdef, V_List_gt, M_list_gt
    
    def deformSrcList_LDDMM(self, v0):
        Dispinv_List_gt, Disp_List_gt, V_List_gt, Phiinv_List_gt, Phi_List_gt, M_list_gt = self.MEpdiff.my_expmap_u2phi(v0=v0)
        Sdef_List_gt = [lm.interp(self.src, Disp) for Disp in Dispinv_List_gt]

        return Sdef_List_gt, Phiinv_List_gt, Phi_List_gt, V_List_gt, M_list_gt


   
    def exec_encoder(self, x):
        # encoder forward pass
        for level, convs in enumerate(self.encoder.encoder):
            for conv in convs:
                x = conv(x)
            x = self.encoder.pooling[level](x)
            b,c,w,h = x.shape
            if self.Config["general"]["sub_version"] in {"F004", "F104"}:
                if h == 8:
                    x = x * self.SME16[...,:8]
                elif h == 16:
                    x = x * self.SME32[...,:16]
                elif h == 32:
                    x = x * self.SME64[...,:32]
        return x
    def exec_encoder_AETT(self, x):
        B,_,_,_ = x.shape
        # encoder forward pass
        for level, convs in enumerate(self.encoder.encoder):
            for conv in convs:
                x = conv(x)
            x = self.encoder.pooling[level](x)
            b,c,w,h = x.shape
            if "AETT" in self.Config["general"]["sub_version"]:
                # if level == 0:
                #     ZS = x
                #     ZS = self.SelfAttention1(ZS) #[160, 4096, 10]
                #     ZS = ZS.permute(0, 2, 1)  #[B,20,256]
                #     ZS = ZS.contiguous().view(B, 10, 64, 64)
                #     x = ZS

                if level == 1:
                    ZS = x
                    ZS = self.SelfAttention2(ZS)
                    ZS = ZS.permute(0, 2, 1)
                    ZS = ZS.contiguous().view(B, 20, 32, 32)
                    x = ZS
                elif level == 2:
                    ZS = x
                    ZS = self.SelfAttention3(ZS)
                    # print(ZS.shape)  #[B,256,20]
                    ZS = ZS.permute(0, 2, 1)  #[B,20,256]
                    ZS = ZS.contiguous().view(B, 40, 16, 8)
                    x = ZS
        
        return x
        

    def exec_remain(self, x):
        for conv in self.remain.remaining:
            x = conv(x)
            if self.Config["general"]["sub_version"] in {"F002", "F003", "S002", "S003", "F001"}:
                x = self.SmoothSignal(x, size=128)
        x = self.remain.flow(x)
        return x
    


    def padding_and_getVsFull(self, ZS):
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/npy2/velocityBeforeIRFFT.npy", ZS.cpu().detach().numpy())

        b,c,_,_ = ZS.shape

        ZF = torch.complex(ZS[:,:int(c/2)], ZS[:,int(c/2):])
        # print("~~~~~~~~~~~~~",torch.max(ZS), torch.min(ZS), torch.mean(ZS))

        out_ft = torch.zeros((ZF.size(0),int(c/2),128,65), dtype=ZF.dtype, device=ZF.device)
        out_ft[:, :, :8, :8] = ZF[:,:, :8, :]
        out_ft[:, :, -8:, :8] = ZF[:,:, -8:, :]

        velocity = torch.fft.irfftn(out_ft, dim=(-2, -1), norm="ortho")
        # print("~~~~~~~~~~~~~",torch.max(velocity), torch.min(velocity), torch.mean(velocity))

        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/npy2/velocityAfterIRFFT.npy", velocity.cpu().detach().numpy())
        return velocity

    

    def forward_S003(self, x): #Input Frequency-S,F  Yes use mlp
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]
        
        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 8]

        b,_,_,_ = ZS.shape
        ZS = ZS.reshape(b, -1)
        ZS = self.mlp(ZS)
        ZS = ZS.reshape(b, 4, 16, 8)  #low dimension features
        ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]
        velocity = self.padding_and_getVsFull(ZS)

        return velocity

    def forward_F003(self, x): #Input Frequency-S,F  Yes use mlp
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]
        
        x = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        x = torch.cat([x.real, x.imag], dim=1)  #[224, 4, 128, 65]
        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 8]

        b,_,_,_ = ZS.shape
        ZS = ZS.reshape(b, -1)
        ZS = self.mlp(ZS)
        ZS = ZS.reshape(b, 4, 16, 8)  #low dimension features
        ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]
        velocity = self.padding_and_getVsFull(ZS)
        return velocity
    
    def forward_S002_SVF_FFT_ABI(self, x): #
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]
        
        if self.Config["general"]["sub_version"].startswith("Cosh"):
            x = self.shearlet(x)
        
        
        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 8]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-Bfmlp-S002.npy", ZS.detach().cpu().numpy())

        b,_,_,_ = ZS.shape
        ZS = ZS.reshape(b, -1)
        ZS = self.mlp(ZS)
        ZS = ZS.reshape(b, 20, 16, 16)  #low dimension features
        
        

        if self.Config["general"]["module_name"] == "July":
            if "UPS" in self.Config["general"]["sub_version"] and not "UPS2" in self.Config["general"]["sub_version"]:
                velocity = self.exec_remain(ZS)
                velocity = self.UPS_ABI(velocity)
            elif "UPS2" in self.Config["general"]["sub_version"]:
                velocity = self.UPS_ABI(ZS)
                velocity = self.exec_remain(velocity)
            else:
                velocity = self.exec_remain(ZS)
                velocity = self.FFT_ABI(velocity)

        return velocity

    def forward_S002(self, x): #Input Frequency-S,F  Yes use mlp
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]
        
        if self.Config["general"]["sub_version"].startswith("Cosh"):
            x = self.shearlet(x)
        
        
        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 8]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-Bfmlp-S002.npy", ZS.detach().cpu().numpy())

        b,_,_,_ = ZS.shape
        ZS = ZS.reshape(b, -1)
        ZS = self.mlp(ZS)
        ZS = ZS.reshape(b, 20, 16, 8)  #low dimension features
        ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]

        #save ZS as numpy
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-S002.npy", ZS.detach().cpu().numpy())
        # assert 3>111

        velocity = self.padding_and_getVsFull(ZS)

        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/FullSpatial-S002.npy", velocity.detach().cpu().numpy())
        # assert 3>111


        velocity = self.exec_remain(velocity)
        return velocity
    
    def shearlet(self, x):
        srcf = x[:,0,...]; tarf = x[:,1,...]
        for xform in self.xforms:
            srcf = xform(srcf)
            tarf = xform(tarf)
        srcf.cat(tarf,dim=1)
        x = torch.cat((srcf.real, srcf.imag), dim=1) #[160, 40, 128, 128]
        return x

    def FFT_ABI(self, x):
        out_1 = x[:,0:1,...]; out_2 = x[:,1:2,...]
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fft2(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fft2(out_2))
        # p3d = (84, 84, 70, 70)
        if out_2.shape[-1] == 32:
            p3d = (48, 48, 48, 48)
        elif out_2.shape[-1] == 16:
            p3d = (56, 56, 56, 56)
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
        # out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
        disp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
        # disp_mf_3 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
        f_xy = torch.cat([disp_mf_1.unsqueeze(1), disp_mf_2.unsqueeze(1)], dim = 1)
        return f_xy
    
    def UPS_ABI(self, x):
        output_tensor_true = F.interpolate(x, size=[128, 128], mode='bilinear', align_corners=True)
        return output_tensor_true
    
    def forward_SATT102_SVF_FFT_ABI(self, x): #
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)

        if self.Config["general"]["sub_version"].startswith("Cosh"):
            x = self.shearlet(x)
            # assert 2>333
        

        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]

        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 16]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-Bfmlp-S002.npy", ZS.detach().cpu().numpy())
        if not "SATT101" in self.Config["general"]["sub_version"]:
            B,_,_,_ = ZS.shape
            ZS = self.SelfAttention(ZS)
            # print(ZS.shape)  #[B,256,20]
            ZS = ZS.permute(0, 2, 1)  #[B,20,256]
            ZS = ZS.contiguous().view(B, 20, 16, 16)  
        else:
            # no attention layer
            pass
        
        
        if "UPS" in self.Config["general"]["sub_version"] and not "UPS2" in self.Config["general"]["sub_version"]:
            velocity = self.exec_remain(ZS)
            velocity = self.UPS_ABI(velocity)
        elif "UPS2" in self.Config["general"]["sub_version"]:
            velocity = self.UPS_ABI(ZS)
            velocity = self.exec_remain(velocity)
        else:
            velocity = self.exec_remain(ZS)
            velocity = self.FFT_ABI(velocity)
            
        return velocity
    
    def forward_SATT_GDN(self, x): #
        print(x.shape)
        ##########################   Encoder  #############################
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)

        if self.Config["general"]["sub_version"].startswith("Cosh"):
            x = self.shearlet(x)
            # assert 2>333
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]

        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 16]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-Bfmlp-S002.npy", ZS.detach().cpu().numpy())
        if not "SATT101" in self.Config["general"]["sub_version"]:
            B,_,_,_ = ZS.shape
            ZS = self.SelfAttention(ZS)
            # print(ZS.shape)  #[B,256,20]
            ZS = ZS.permute(0, 2, 1)  #[B,20,256]
            ZS = ZS.contiguous().view(B, 20, 16, 16)  
        else:
            # no attention layer
            pass
        
        ##########################   Learn Geodesic  #############################
        L_Z0 = ZS; L_Z = ZS
        L_Z_List = [L_Z0]


        for t in range(1,self.TSteps): #TSteps=7 V0: V1~V6      Predict 6 steps
            L_Z= self.model_v(L_Z)
            L_Z_List.append(L_Z)

        ##########################   Upsample  #############################
        V_List = []
        for t in range(0, self.TSteps):
            ZS = L_Z_List[t]
            if "UPS" in self.Config["general"]["sub_version"] and not "UPS2" in self.Config["general"]["sub_version"]:
                velocity = self.exec_remain(ZS)
                velocity = self.UPS_ABI(velocity)
            elif "UPS2" in self.Config["general"]["sub_version"]:
                velocity = self.UPS_ABI(ZS)
                velocity = self.exec_remain(velocity)
            else:
                velocity = self.exec_remain(ZS)
                velocity = self.FFT_ABI(velocity)
            V_List.append(velocity)
        return V_List
    
    def forward_SATT_Numerical(self, x): #
        print(x.shape)
        ##########################   Encoder  #############################
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)

        if self.Config["general"]["sub_version"].startswith("Cosh"):
            x = self.shearlet(x)
            # assert 2>333
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]

        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 16]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-Bfmlp-S002.npy", ZS.detach().cpu().numpy())
        if not "SATT101" in self.Config["general"]["sub_version"]:
            B,_,_,_ = ZS.shape
            ZS = self.SelfAttention(ZS)
            # print(ZS.shape)  #[B,256,20]
            ZS = ZS.permute(0, 2, 1)  #[B,20,256]
            ZS = ZS.contiguous().view(B, 20, 16, 16)  
        else:
            # no attention layer
            pass
        
        
        if "UPS" in self.Config["general"]["sub_version"] and not "UPS2" in self.Config["general"]["sub_version"]:
            velocity = self.exec_remain(ZS)
            velocity = self.UPS_ABI(velocity)
        elif "UPS2" in self.Config["general"]["sub_version"]:
            velocity = self.UPS_ABI(ZS)
            velocity = self.exec_remain(velocity)
        else:
            velocity = self.exec_remain(ZS)
            velocity = self.FFT_ABI(velocity)
        return velocity
    
    def forward_SATT_GDN_DifuS(self, x): #
        # print(x.shape)
        ##########################   Encoder  #############################
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]

        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 16]
        ##########################   Learn Geodesic  #############################
        L_Z0 = ZS; L_Z = ZS
        L_Z_List = [L_Z0]

        for t in range(1,self.TSteps): #TSteps=7 V0: V1~V6      Predict 6 steps
            L_Z= self.model_v(L_Z)
            L_Z_List.append(L_Z)

        return L_Z_List







    def forward_GDN_with_V0(self, Z0):
        print(Z0.shape)  #[16, 20, 16, 16]
        L_Z_List = [Z0]
        L_Z = Z0
        for t in range(1,self.TSteps): #TSteps=7 V0: V1~V6      Predict 6 steps
            L_Z= self.model_v(L_Z)
            L_Z_List.append(L_Z)

        V_List = []
        for ZS in L_Z_List:  #sub_version="SATT101_UPS2"
            if "UPS" in self.Config["general"]["sub_version"] and not "UPS2" in self.Config["general"]["sub_version"]:
                assert 1>123
                velocity = self.exec_remain(ZS)
                velocity = self.UPS_ABI(velocity)
            elif "UPS2" in self.Config["general"]["sub_version"]:
                assert 1>125
                velocity = self.UPS_ABI(ZS)
                velocity = self.exec_remain(velocity)
            else:
                assert 1>120
                velocity = self.exec_remain(ZS)
                velocity = self.FFT_ABI(velocity)
            V_List.append(velocity)
        
        return V_List



    def forward_SATT102(self, x): #Input Frequency-S,F  Yes use mlp
        print(x.shape)
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)

        if self.Config["general"]["sub_version"].startswith("Cosh"):
            x = self.shearlet(x)
            # assert 2>333
        

        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]

        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 8]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-Bfmlp-S002.npy", ZS.detach().cpu().numpy())

        B,_,_,_ = ZS.shape
        ZS = self.SelfAttention(ZS)
        # print(ZS.shape)  #[B,256,20]
        ZS = ZS.permute(0, 2, 1)  #[B,20,256]
        ZS = ZS.contiguous().view(B, 40, 16, 8)


        ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]

        #save ZS as numpy
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-S002.npy", ZS.detach().cpu().numpy())
        # assert 3>111

        velocity = self.padding_and_getVsFull(ZS)

        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/FullSpatial-S002.npy", velocity.detach().cpu().numpy())
        # assert 3>111


        velocity = self.exec_remain(velocity)
        return velocity
    

    def forward_SAETT102(self, x): #Input Frequency-S,F  Yes use mlp
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]
        
        
        ZS = self.exec_encoder_AETT(x)   #low dimension features  #[224, 20, 16, 8]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-Bfmlp-S002.npy", ZS.detach().cpu().numpy())

        ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]

        #save ZS as numpy
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-S002.npy", ZS.detach().cpu().numpy())
        # assert 3>111

        velocity = self.padding_and_getVsFull(ZS)

        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/FullSpatial-S002.npy", velocity.detach().cpu().numpy())
        # assert 3>111
        velocity = self.exec_remain(velocity)
        return velocity
    
    
    def forward_F002(self, x): #Input Frequency-S,F  Yes use mlp
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]
        
        x = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        x = torch.cat([x.real, x.imag], dim=1)  #[224, 4, 128, 65]
        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 8]

        b,_,_,_ = ZS.shape
        ZS = ZS.reshape(b, -1)
        ZS = self.mlp(ZS)
        ZS = ZS.reshape(b, 20, 16, 8)  #low dimension features
        ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]

        #save ZS as numpy
        np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-F002.npy", ZS.detach().cpu().numpy())
        assert 3>111

        velocity = self.padding_and_getVsFull(ZS)

        velocity = self.exec_remain(velocity)
        return velocity


    def forward_FATT102(self, x): #Input Frequency-S,F  Yes use mlp
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]
        
        x = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        x = torch.cat([x.real, x.imag], dim=1)  #[224, 4, 128, 65]
        ZS = self.exec_encoder(x)   #low dimension features  #[224, 20, 16, 8]

        B,_,_,_ = ZS.shape
        ZS = self.SelfAttention(ZS)
        # print(ZS.shape)  #[B,256,20]
        ZS = ZS.permute(0, 2, 1)  #[B,20,256]
        ZS = ZS.contiguous().view(B, 20, 16, 8)
       


        ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]

        #save self.SME16 as numpy
        np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/SME16.npy", self.SME16.detach().cpu().numpy())
        assert 3>111

        #save ZS as numpy
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-F002.npy", ZS.detach().cpu().numpy())
        # assert 3>111

        velocity = self.padding_and_getVsFull(ZS)

        velocity = self.exec_remain(velocity)
        return velocity


    def forward_F001(self, x): #Input Frequency-S,F  Not use mlp
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        b,c,w,h = x.shape
        self.src = x[:,0:int(c/2),...] #[32, 1, 64, 64]
        self.tar = x[:,int(c/2):,...] #[32, 1, 64, 64]

        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/npy2/ST_BeforeIRFFT.npy", x.cpu().detach().numpy())
        x = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho") #[224, 2, 128, 65]
        x = torch.cat([x.real, x.imag], dim=1)  #[224, 4, 128, 65]
        # np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/npy2/ST_AfterIRFFT.npy", x.cpu().detach().numpy())
        ZS = self.exec_encoder(x)   #low dimension features
        # ZS = ZS * self.SME16[...,:8]  #[224, 20, 16, 8]
        # save the ZS as numpy
        np.save("/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/July/LowF-F001.npy", ZS.detach().cpu().numpy())
        assert 3>111
        
        velocity = self.padding_and_getVsFull(ZS)
        velocity = self.exec_remain(velocity)
        return velocity



    def forward_given_V_Z_List(self, V_Z_List):
        ##########################   Upsample  #############################
        V_List = []
        for ZS in V_Z_List:
            if "UPS" in self.Config["general"]["sub_version"] and not "UPS2" in self.Config["general"]["sub_version"]:
                velocity = self.exec_remain(ZS)
                velocity = self.UPS_ABI(velocity)
            elif "UPS2" in self.Config["general"]["sub_version"]:
                velocity = self.UPS_ABI(ZS)
                velocity = self.exec_remain(velocity)
            else:
                velocity = self.exec_remain(ZS)
                velocity = self.FFT_ABI(velocity)
            V_List.append(velocity)
        return V_List
        

    def forward(self, x, y=None,SrcTar=None,altarFlag="Joint",srcRGB=None):  #altarFlag="Register" #"Joint" "Fno"   sub_version="SATT101_UPS2"
        if self.Config["general"]["module_name"] in {"JulyGDN_DifuS"}:
            if self.Config["general"]["mode"] == "train":
                V_Z_List = self.forward_SATT_GDN_DifuS(x) #[[b,20,16,16],...] 7
                if self.Config["general"]["DifuS_version"] == "way1":
                    data = torch.stack(V_Z_List, dim=2) #[b,20,7,16,16]
                elif self.Config["general"]["DifuS_version"] == "way2":
                    data = torch.stack(V_Z_List, dim=1) #[b,7,20,16,16]
                return data


            elif self.Config["general"]["mode"] == "test":
                V_Z_List_Encoder = self.forward_SATT_GDN_DifuS(SrcTar) #[[b,20,16,16],...] 7
                if self.Config["general"]["DifuS_version"] == "way1":
                    V_Z_List_Encoder = torch.stack(V_Z_List_Encoder, dim=2) #[b,20,7,16,16]
                elif self.Config["general"]["DifuS_version"] == "way2":
                    V_Z_List_Encoder = torch.stack(V_Z_List_Encoder, dim=1) #[b,7,20,16,16]
                #V_Z_List src_cond
                V_Z_List = x
                src_cond = y
                self.src = src_cond[:,0:1,...]
                
                V_List = self.forward_given_V_Z_List(V_Z_List)
                # print("V_List", len(V_List), V_List[0].shape)   ###V_List 7 torch.Size([16, 2, 128, 128])
                V_List_Dec = self.forward_GDN_with_V0(V_Z_List[0])
                
                if srcRGB is not None:
                    Sdef_List, Phiinv_List, Phi_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt, srcRGB_List, srcRGB_List_gt = self.deformSrcList_GDN(V_List, srcRGB = srcRGB)
                    Sdef_List_Dec, _, _, _, _, _, _, _, _, _, srcRGB_List_Dec, _ = self.deformSrcList_GDN(V_List_Dec, srcRGB = srcRGB)
                    return Sdef_List, Phiinv_List, Phi_List, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt, V_Z_List_Encoder,srcRGB_List, srcRGB_List_gt, V_List_Dec, Sdef_List_Dec, srcRGB_List_Dec
                else:
                    Sdef_List, Phiinv_List, Phi_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt = self.deformSrcList_GDN(V_List)
                    Sdef_List_Dec, _, _, _, _, _, _, _, _, _ = self.deformSrcList_GDN(V_List_Dec)
                    return Sdef_List, Phiinv_List, Phi_List, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt, V_Z_List_Encoder, V_List_Dec, Sdef_List_Dec

                