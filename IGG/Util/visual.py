import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import matplotlib.font_manager as fm
from scipy.ndimage import zoom
from Util.utils import Mgridplot
import SimpleITK as sitk
import os
import random


def get_v_img(velocity, scale = 3):
    velocity = zoom(velocity, (1, scale, scale), order=3)  #(1, 32, 32, 3)
    velocity = (velocity-np.min(velocity))/(np.max(velocity)-np.min(velocity))*255
    a1=velocity[0,...];a2=velocity[1,...];a3=np.zeros_like(a1)
    r = Image.fromarray(a1).convert('L');g = Image.fromarray(a2).convert('L');b = Image.fromarray(a3).convert('L')
    velocity = Image.merge('RGB',(r,g,b))
    return velocity
def get_disp_img(velocity, scale = 3):
    velocity = zoom(velocity, (1, scale, scale), order=3)  #(1, 32, 32, 3)
    velocity = (velocity-np.min(velocity))/(np.max(velocity)-np.min(velocity))*255
    a1=velocity[0,...];a2=velocity[1,...];a3=np.zeros_like(a1)
    r = Image.fromarray(a1).convert('L');g = Image.fromarray(a2).convert('L');b = Image.fromarray(a3).convert('L')
    velocity = Image.merge('RGB',(r,g,b))
    return velocity

def get_vfft_img(velocity, scale = 3):
    v = np.fft.fftn(velocity, axes=(-2, -1), norm="forward")
    velocity = np.fft.fftshift(v, axes=(-2, -1))

    # print(np.max(v.real), np.min(v.real), np.max(v.imag), np.min(v.imag))
    # 0.1218468933798249 -0.037820517155068956 0.23604003402922302 -0.23604003402922302
    Real = velocity.real; Real = Real.astype(np.float32)
    Imag = velocity.imag; Imag = Imag.astype(np.float32)


    Real_ = zoom(Real[0], (scale, scale), order=3)*255; velocityR = Image.fromarray(Real_, mode='F'); 
    Imag_ = zoom(Imag[0], (scale, scale), order=3)*255; velocityI = Image.fromarray(Imag_, mode='F');


    return [velocityR, np.max(Real), np.min(Real)], [velocityI, np.max(Imag), np.min(Imag)]

def get_phi_img(phi, scale = 3, temp="", imagesize=64): 
    # print(phi.shape) #(2, 128, 128)
    if phi.shape[-1] == 2:
        phi_t = np.transpose(phi,[2,0,1]) 
    else:
        phi_t = phi
    phi_t = np.expand_dims(phi_t,axis=0)
    Mgridplot(phi_t, temp, int(imagesize/2), int(imagesize/2), False,  dpi=imagesize, scale=scale) 
    PDphi_t = sitk.GetArrayFromImage(sitk.ReadImage(temp))
    
    GTphiinv = PDphi_t.astype(np.float32)
    a1=GTphiinv[...,0]
    a2=GTphiinv[...,1]
    a3=GTphiinv[...,2]
    r = Image.fromarray(a1).convert('L')
    g = Image.fromarray(a2).convert('L')
    b = Image.fromarray(a3).convert('L')
    GTphiinv = Image.merge('RGB',(r,g,b))
    GTphiinv = GTphiinv.convert('RGB')

    os.remove(temp)

    return GTphiinv 

def get_phi_img_arr(phi_list, scale = 3, CNum=1, imagesize=128, Config=None): 
    # return 123
    # print(phi_list[0].shape) #(2, 128, 128)
    visjpg = Config["general"]["visjpg"]
    

    temp = f"{visjpg}/temp"
    #generate a random file name
    # temp = temp + "temp{}.jpg"
    temp = "temp{}.jpg".format(random.randint(1, 99999))

    res = []
    for cnt in range(CNum):
        phi = phi_list[cnt].detach().cpu().numpy()
        if phi.shape[-1] == 2:
            phi_t = np.transpose(phi,[2,0,1]) 
        else:
            phi_t = phi
        phi_t = np.expand_dims(phi_t,axis=0)
        Mgridplot(phi_t, temp, int(imagesize/2), int(imagesize/2), False,  dpi=imagesize, scale=scale) 
        PDphi_t = sitk.GetArrayFromImage(sitk.ReadImage(temp))  #(64, 64, 3)
       
        PDphi_t = PDphi_t.transpose(2,0,1)  #(64, 64, 3) -> (3, 64, 64)
        # normalize to 0,1
        PDphi_t = (PDphi_t - np.min(PDphi_t))/(np.max(PDphi_t) - np.min(PDphi_t))
        os.remove(temp)
        res.append(torch.from_numpy(PDphi_t))
    return torch.stack(res, dim=0) #(CNum, 3, 64, 64)
   


def Save_intermediate_resust_as_numpy(Config, model=None):
    visjpg = Config["general"]["visjpg"]
    temp = f"{visjpg}/temp"
    temp = temp + "temp.jpg"

    if model:
        saving_path = f"{visjpg}/Predicted_Intermediate.npy"
        np.save(saving_path, model.record_of_latentfeatures)
        print(f"save done {saving_path} \n")



def draw_one_picture(test_result, Config, scale = 3, num_per_picture = 20, shootingType="svf", model=None, TSteps=None):
    ImgSize = Config["DataSet"]["test_img_size"]
    gap = ImgSize*scale
    TSteps = TSteps or Config["general"]["num_steps"]
    visjpg = Config["general"]["visjpg"]
    temp = f"{visjpg}/temp"
    temp = temp + "temp.jpg"
    

    print("TSteps: ", TSteps, visjpg)
    # 7 /home

    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),10*scale)

    # print(len(test_result['src'])) #
    # print(test_result['src'][0].shape) #[320, 1, 128, 128]
    # assert 3>333
    if test_result['src'][0].shape[0] >1 and len(test_result['src'])==1:
        srcList = test_result['src'][0] #32,1,128,128
        tarList = test_result['tar'][0]
        SdefListList = test_result['Sdef'][0]
        phiinvListList = test_result['phiinv'][0]
        phiListList = test_result['phi'][0]
        velocityListList = test_result['velocity'][0]

        ''' print(type(SdefListList), type(phiinvListList), type(phiListList), type(velocityListList))
                # <class 'list'>    <class 'list'>      <class 'list'>      <class 'torch.Tensor'>
        print(len(SdefListList), len(phiinvListList), len(phiListList)) #7 7 7 '''
        
        ##  convert the 32,1,128,128 to list
        srcList = [srcList[i].unsqueeze(0) for i in range(srcList.shape[0])]
        tarList = [tarList[i].unsqueeze(0) for i in range(tarList.shape[0])]
        velocityListList = [velocityListList[i].unsqueeze(0) for i in range(velocityListList.shape[0])]


        SdefListList_new = []
        phiinvListList_new = []
        phiListList_new = []

        
        for i in range(SdefListList[0].shape[0]):
            SdefListList_new.append(torch.stack([SdefListList[j][i].unsqueeze(0) for j in range(len(SdefListList))], dim=0))
            phiinvListList_new.append(torch.stack([phiinvListList[j][i].unsqueeze(0) for j in range(len(phiinvListList))], dim=0))
            phiListList_new.append(torch.stack([phiListList[j][i].unsqueeze(0) for j in range(len(phiListList))], dim=0))

        SdefListList = SdefListList_new
        phiinvListList = phiinvListList_new
        phiListList = phiListList_new

    else:
        srcList = test_result['src']
        tarList = test_result['tar']
        SdefListList = test_result['Sdef']
        phiinvListList = test_result['phiinv']
        phiListList = test_result['phi']
        velocityListList = test_result['velocity']
        
    
    
    
    cnt = 0
    TotalNum = min(len(srcList), 32)
    for i in range(TotalNum):
        cnt += 1
        if i%num_per_picture == 0:
            res = Image.new('RGB', (gap*(max(TSteps+1, 4)),gap*5*num_per_picture),'white')  #宽   高   size/size    gap/gap   40/9
            draw = ImageDraw.Draw(res)

        src_ = srcList[i].detach().cpu().numpy()
        # print(src_.shape)
        # assert 3>111



        tar_ = tarList[i].detach().cpu().numpy()
        Sdef_ = SdefListList[i][-1].detach().cpu().numpy()
        Diff_ = np.abs(tar_ - Sdef_)

        SdefList = SdefListList[i]
        phiinvList = phiinvListList[i]
        phiList = phiListList[i]
        velocityList = velocityListList[i]
        
            

        baseH = (i%num_per_picture)*gap*5

        # print("visual.py", src_.shape, tar_.shape, Sdef_.shape, TotalNum)
        # assert 3>333
        # (1, 1, 64, 64) (1, 1, 64, 64) (1, 1, 64, 64) 10
        index = 0

        
        # print(src_[index][0].shape, tar_[index][0].shape, Sdef_[index][0].shape, Diff_[index][0].shape)
        # assert 3>222
        src_ = zoom(src_[index][0], (scale, scale), order=3)*255; srcImg = Image.fromarray(src_, mode='F'); res.paste(srcImg, box=(0,baseH));
        draw.text((0,baseH),f"{cnt}_Src",font=font,fill=(255,255,255))
        tar_ = zoom(tar_[index][0], (scale, scale), order=3)*255; tarImg = Image.fromarray(tar_, mode='F'); res.paste(tarImg, box=(gap,baseH));
        draw.text((gap,baseH),"Tar",font=font,fill=(255,255,255))
        Sdef_ = zoom(Sdef_[index][0], (scale, scale), order=3)*255; SdefImg = Image.fromarray(Sdef_, mode='F'); res.paste(SdefImg, box=(2*gap,baseH));
        draw.text((2*gap,baseH),"Deformed",font=font,fill=(255,255,255))
        DiffImg = zoom(Diff_[index][0], (scale, scale), order=3)*255; DiffImg = Image.fromarray(DiffImg, mode='F'); res.paste(DiffImg, box=(3*gap,baseH));
        draw.text((3*gap,baseH),"Diff(tar-dfm)",font=font,fill=(255,255,255))

        # print("visual.py",velocityList.shape, phiinvList[0].shape, SdefList[0].shape)
        #torch.Size([10, 2, 128, 128]) torch.Size([10, 128, 128, 2]) torch.Size([10, 1, 128, 128])
        
        if shootingType == "svf":
            VImg = get_v_img(velocityList[index].detach().cpu().numpy(), scale = scale); 
            res.paste(VImg, box=(0,baseH+gap*2));draw.text((0,baseH+gap*2),"V0",font=font,fill=(255,255,255))
            VFFTImgReal,  VFFTImgImag= get_vfft_img(velocityList[index].detach().cpu().numpy(), scale = scale); 
            res.paste(VFFTImgReal[0], box=(gap,baseH+gap*2)); draw.text((gap,baseH+gap*2),f"FFTReal{VFFTImgReal[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))
            res.paste(VFFTImgImag[0], box=(2*gap,baseH+gap*2)); draw.text((2*gap,baseH+gap*2),f"FFTImag{VFFTImgImag[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))
            
        
            
        # print(TSteps, len(SdefList), len(phiinvList), len(phiList))
        # 7 8 8 8

        for j in range(TSteps):
            Sdef_ = zoom(SdefList[j][index][0].detach().cpu().numpy(), (scale, scale), order=3)*255; SdefImg = Image.fromarray(Sdef_, mode='F'); res.paste(SdefImg, box=(j*gap,baseH+gap));
            draw.text((j*gap,baseH+gap),f"Deformed{j}",font=font,fill=(255,255,255))
            
            # print(velocityList[j].shape, phiinvList[j].shape, phiList[j].shape)
            # torch.Size([2, 64, 64]) torch.Size([1, 64, 64, 2]) torch.Size([1, 64, 64, 2])
            
            PhiinvImg = get_phi_img(phiinvList[j][index].detach().cpu().numpy(), scale = scale, temp = temp, imagesize = ImgSize); res.paste(PhiinvImg, box=(j*gap,baseH+gap*3))
            draw.text((j*gap,baseH+gap*3),f"Phiinv{j}",font=font,fill=(255,0,0))

            PhiImg = get_phi_img(phiList[j][index].detach().cpu().numpy(), scale = scale, temp = temp, imagesize = ImgSize); res.paste(PhiImg, box=(j*gap,baseH+gap*4))
            draw.text((j*gap,baseH+gap*4),f"Phi{j}",font=font,fill=(255,0,0))

            if shootingType == "lddmm":
                VImg = get_v_img(velocityList[j][index].detach().cpu().numpy(), scale = scale); 
                res.paste(VImg, box=(j*gap,baseH+gap*2));draw.text((j*gap,baseH+gap*2),"V0",font=font,fill=(255,255,255))

                
                # VFFTImgReal,  VFFTImgImag= get_vfft_img(velocityList[0].detach().cpu().numpy(), scale = scale); 
                # res.paste(VFFTImgReal[0], box=(gap,baseH+gap*2)); draw.text((gap,baseH+gap*2),f"FFTReal{VFFTImgReal[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))
                # res.paste(VFFTImgImag[0], box=(2*gap,baseH+gap*2)); draw.text((2*gap,baseH+gap*2),f"FFTImag{VFFTImgImag[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))


        if (i+1)%num_per_picture == 0 or (i+1) == TotalNum:
            saving_path = f"{visjpg}/{i//num_per_picture}.jpg"
            res.save(saving_path)
            print(f"save done {saving_path} \n")


        # if model:
        #     saving_path = f"{visjpg}/{i//num_per_picture}.npy"
        #     np.save(saving_path, model.record_of_latentfeatures)
        #     print(f"save done {saving_path} \n")




def draw_one_picture2(test_result, Config, scale = 3, TotalNum=4, Snum=7, num_per_picture = 20, shootingType="lddmm", model=None, TSteps=None):
    ImgSize = Config["DataSet"]["test_img_size"]
    gap = ImgSize*scale
    TSteps = TSteps or Config["general"]["num_steps"]
    visjpg = Config["general"]["visjpg"]
    temp = f"{visjpg}/temp"
    temp = temp + "temp.jpg"
    

    print("TSteps: ", TSteps, visjpg)
    # 7 /home

    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),10*scale)

    # print(len(test_result['src'])) #
    # print(test_result['src'][0].shape) #[320, 1, 128, 128]
    # assert 3>333
    
    srcList = test_result['src']
    tarList = test_result['tar']
    SdefListList = test_result['Sdef']
    phiinvListList = test_result['phiinv']
    phiListList = test_result['phi']
    velocityListList = test_result['velocity']

    velocityListList_GT = test_result['VList_gt']
    phiinvListList_GT = test_result['Phiinv_gt']
    SdefListList_GT = test_result['Sdef_gt']
    DispinvListList = test_result['Dispinv']
    print("@@@@@@@@@@@@@@@@@@@@@@@",len(srcList))
    
    #num_per_picture=20
    cnt = 0
    for i in range(min(TotalNum, len(srcList))):
        cnt += 1
        if (i*Snum)%num_per_picture == 0: #:  *Snum because digits 0-9
            res = Image.new('RGB', (gap*(max(TSteps, 4)),gap*8*num_per_picture),'white')  #宽   高   size/size    gap/gap   40/9
            draw = ImageDraw.Draw(res)

        src_ = srcList[i].detach().cpu().numpy()
        # print(src_.shape)
        # assert 3>111



        tar_ = tarList[i].detach().cpu().numpy()
        Sdef_ = SdefListList[i][-1].detach().cpu().numpy()
        Diff_ = np.abs(tar_ - Sdef_)

        SdefList = SdefListList[i]
        phiinvList = phiinvListList[i]
        phiList = phiListList[i]
        velocityList = velocityListList[i]
        velocityList_GT = velocityListList_GT[i]
        phiinvList_GT = phiinvListList_GT[i]
        SdefList_GT = SdefListList_GT[i]
        DispinvList = DispinvListList[i]
            

        

        print("visual.py", src_.shape, tar_.shape, Sdef_.shape, TotalNum)
        # assert 3>333
        # visual.py (10, 1, 128, 128) (10, 1, 128, 128) (10, 1, 128, 128) 2

        for index in range(Snum):
            baseH = (((i*Snum)%num_per_picture)+index)*gap*8
            
            # index = 0
            print("@@@@@@@@@@@@@@@@@@@@@@@",len(srcList), index)
            #(128, 128) (128, 128) (128, 128) (128, 128)
            print(src_[index][0].shape, tar_[index][0].shape, Sdef_[index][0].shape, Diff_[index][0].shape)
            # assert 3>222
            srcImg = zoom(src_[index][0], (scale, scale), order=3)*255; srcImg = Image.fromarray(srcImg, mode='F'); res.paste(srcImg, box=(0,baseH));
            draw.text((0,baseH),f"{cnt}_Src",font=font,fill=(255,255,255))
            tarImg = zoom(tar_[index][0], (scale, scale), order=3)*255; tarImg = Image.fromarray(tarImg, mode='F'); res.paste(tarImg, box=(gap,baseH));
            draw.text((gap,baseH),"Tar",font=font,fill=(255,255,255))
            SdefImg = zoom(Sdef_[index][0], (scale, scale), order=3)*255; SdefImg = Image.fromarray(SdefImg, mode='F'); res.paste(SdefImg, box=(2*gap,baseH));
            draw.text((2*gap,baseH),"Deformed",font=font,fill=(255,255,255))
            DiffImg = zoom(Diff_[index][0], (scale, scale), order=3)*255; DiffImg = Image.fromarray(DiffImg, mode='F'); res.paste(DiffImg, box=(3*gap,baseH));
            draw.text((3*gap,baseH),"Diff(tar-dfm)",font=font,fill=(255,255,255))

            # print("visual.py",velocityList.shape, phiinvList[0].shape, SdefList[0].shape)
            #torch.Size([10, 2, 128, 128]) torch.Size([10, 128, 128, 2]) torch.Size([10, 1, 128, 128])
            
            if shootingType == "svf":
                VImg = get_v_img(velocityList[index].detach().cpu().numpy(), scale = scale); 
                res.paste(VImg, box=(0,baseH+gap*2));draw.text((0,baseH+gap*2),"V0",font=font,fill=(255,255,255))
                VFFTImgReal,  VFFTImgImag= get_vfft_img(velocityList[index].detach().cpu().numpy(), scale = scale); 
                res.paste(VFFTImgReal[0], box=(gap,baseH+gap*2)); draw.text((gap,baseH+gap*2),f"FFTReal{VFFTImgReal[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))
                res.paste(VFFTImgImag[0], box=(2*gap,baseH+gap*2)); draw.text((2*gap,baseH+gap*2),f"FFTImag{VFFTImgImag[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))
                
            
                
            # print(TSteps, len(SdefList), len(phiinvList), len(phiList))
            # 7 8 8 8

            for j in range(TSteps):
                SdefImg = zoom(SdefList[j][index][0].detach().cpu().numpy(), (scale, scale), order=3)*255; SdefImg = Image.fromarray(SdefImg, mode='F'); res.paste(SdefImg, box=(j*gap,baseH+gap));
                draw.text((j*gap,baseH+gap),f"Deformed{j}",font=font,fill=(255,255,255))
                
                # print(velocityList[j].shape, phiinvList[j].shape, phiList[j].shape)
                # # torch.Size([2, 64, 64]) torch.Size([1, 64, 64, 2]) torch.Size([1, 64, 64, 2])
                # print(phiinvList[j][index].shape) #torch.Size([2, 128, 128])
                # assert 3>333
                d = torch.stack((phiinvList[j][index][1], phiinvList[j][index][0]), dim=0)
                # d = phiinvList[j][index]
                PhiinvImg = get_phi_img(d.detach().cpu().numpy(), scale = scale, temp = temp, imagesize = ImgSize); res.paste(PhiinvImg, box=(j*gap,baseH+gap*3))
                draw.text((j*gap,baseH+gap*3),f"Phiinv{j}",font=font,fill=(255,0,0))


                # DispinvImg = get_disp_img(DispinvList[j][index].detach().cpu().numpy(), scale = scale);  
                # res.paste(DispinvImg, box=(j*gap,baseH+gap*3));draw.text((j*gap,baseH+gap*3),"Dispinv",font=font,fill=(255,255,255))

                d = torch.stack((phiList[j][index][1], phiList[j][index][0]), dim=0)
                PhiImg = get_phi_img(d.detach().cpu().numpy(), scale = scale, temp = temp, imagesize = ImgSize); res.paste(PhiImg, box=(j*gap,baseH+gap*4))
                draw.text((j*gap,baseH+gap*4),f"Phi{j}",font=font,fill=(255,0,0))

                if shootingType == "lddmm": ### For GDN
                    VImg = get_v_img(velocityList[j][index].detach().cpu().numpy(), scale = scale); 
                    #save VImg as png
                    res.paste(VImg, box=(j*gap,baseH+gap*2));draw.text((j*gap,baseH+gap*2),f"V{j}",font=font,fill=(255,255,255))

                    VImg_GT = get_v_img(velocityList_GT[j][index].detach().cpu().numpy(), scale = scale);
                    res.paste(VImg_GT, box=(j*gap,baseH+gap*5));draw.text((j*gap,baseH+gap*5),f"V_GT{j}",font=font,fill=(255,255,255))

                    d = torch.stack((phiinvList_GT[j][index][1], phiinvList_GT[j][index][0]), dim=0)
                    PhiinvImg_GT = get_phi_img(d.detach().cpu().numpy(), scale = scale, temp = temp, imagesize = ImgSize); res.paste(PhiinvImg_GT, box=(j*gap,baseH+gap*6))
                    draw.text((j*gap,baseH+gap*6),f"Phiinv_GT{j}",font=font,fill=(255,0,0))

                    SdefImg_GT = zoom(SdefList_GT[j][index][0].detach().cpu().numpy(), (scale, scale), order=3)*255; SdefImg_GT = Image.fromarray(SdefImg_GT, mode='F'); res.paste(SdefImg_GT, box=(j*gap,baseH+gap*7));
                    draw.text((j*gap,baseH+gap*7),"Deformed_GT",font=font,fill=(255,255,255))


                    VImg.save(f"{visjpg}/V{j}_{i}.png")
                    VImg_GT.save(f"{visjpg}/V_GT{j}_{i}.png")
                    PhiinvImg.save(f"{visjpg}/Phiinv{j}_{i}.png")
                    # SdefImg.save(f"{visjpg}/Sdef_GT{j}_{i}.png")
                    PhiImg.save(f"{visjpg}/Phi{j}_{i}.png")

                    
                    # VFFTImgReal,  VFFTImgImag= get_vfft_img(velocityList[0].detach().cpu().numpy(), scale = scale); 
                    # res.paste(VFFTImgReal[0], box=(gap,baseH+gap*2)); draw.text((gap,baseH+gap*2),f"FFTReal{VFFTImgReal[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))
                    # res.paste(VFFTImgImag[0], box=(2*gap,baseH+gap*2)); draw.text((2*gap,baseH+gap*2),f"FFTImag{VFFTImgImag[1]:.2f}  {VFFTImgReal[2]:.2f}",font=font,fill=(255,255,255))


        if ((i+1)*Snum) %num_per_picture == 0 or (i+1) == TotalNum:
            saving_path = f"{visjpg}/{(i*Snum)//num_per_picture}.jpg"
            res.save(saving_path)
            print(f"save done {saving_path} \n")


            # if model:
            #     saving_path = f"{visjpg}/{i//num_per_picture}.npy"
            #     np.save(saving_path, model.record_of_latentfeatures)
            #     print(f"save done {saving_path} \n")

