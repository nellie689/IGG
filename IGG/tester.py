import os
import argparse
import logging
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from datasets.loaddata import loadTrain, loadTest
from tqdm import tqdm
from timeit import default_timer
from Util.utils import LpLoss, get_source_gradient, video_tensor_to_gif, video_tensor_to_mp4, RGB2GRAY, get_text_list, tensor_to_image, getReductedData, get_text_list_from_time, FID, KID
from Util.EpdiffLib import Grad
from PIL import Image,ImageDraw,ImageFont
import matplotlib.font_manager as fm
from scipy.ndimage import zoom
from Util.visual import draw_one_picture, Save_intermediate_resust_as_numpy, draw_one_picture2, get_phi_img_arr
import torch.profiler
from einops import rearrange
from torch.profiler import profile, record_function, ProfilerActivity
import SimpleITK as sitk


criterion = nn.MSELoss()
myloss = LpLoss(size_average=False)
regloss = Grad('l2').loss

def tester_GDN(Config, model):
    altarFlag = "Register"
    altarFlag = "Fno"
    
    dataName = Config["DataSet"]["dataName"]
    visjpgpath = Config["general"]["visjpg"]
    textpath = f"{visjpgpath}/text"
    textf_GDN_DiFuS_path = f"{textpath}/reg_{dataName}_GDN_DiFuS.txt"
    textf_EPDiff_path = f"{textpath}/reg_{dataName}_EPDiff.txt"
    if not os.path.exists(textpath):
        os.makedirs(textpath)
    textf_GDN_DiFuS = open(textf_GDN_DiFuS_path, 'w')
    textf_EPDiff = open(textf_EPDiff_path, 'w')


    random.seed(Config["general"]["seed"])
    np.random.seed(Config["general"]["seed"])
    torch.manual_seed(Config["general"]["seed"])
    torch.cuda.manual_seed(Config["general"]["seed"])

    WSimi = Config["LossWeightSvf"]["WSimi"]
    WReg = Config["LossWeightSvf"]["WReg"]
    snapshot_path = Config["general"]["snapshot_path"]
    loadModelAtBegin = Config["general"]["loadModelAtBegin"]
    

    loadModelAtBegin = "Yes"
    if loadModelAtBegin == "Yes":
        pass
    else:
        epoch_num = Config["LStrategy"]["max_epochs"] - 1
        if altarFlag == "Joint":
            path_model_joint = snapshot_path+ f'/registration_SepFNODeCoder{Config["general"]["SepFNODeCoder"]}_{epoch_num}.pth'
            #load the model
            checkpoint = torch.load(path_model_joint)
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.remain.load_state_dict(checkpoint['remain'])
            if Config["general"]["SepFNODeCoder"] == "Yes":
                model.model_v.load_state_dict(checkpoint['model_v'])
                model.remain_model_v.load_state_dict(checkpoint['remain_model_v'])
            elif Config["general"]["SepFNODeCoder"] == "No":
                model.model_v.load_state_dict(checkpoint['model'])
        elif altarFlag == "Register" or altarFlag == "Fno":
            path_model_Register = snapshot_path+ "/Register_{}.pth".format(epoch_num)
            path_model_ATT = snapshot_path+ "/ATT_{}.pth".format(epoch_num)
            #load the model
            checkpoint = torch.load(path_model_Register)
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.remain.load_state_dict(checkpoint['remain'])
            # model.SelfAttention.load_state_dict(checkpoint['SelfAttention'])
            if hasattr(model, 'SelfAttention'):
                checkpoint = torch.load(path_model_ATT)
                model.SelfAttention.load_state_dict(checkpoint['SelfAttention'])

        if altarFlag == "Fno":
            ModesFno = Config["general"]["ModesFno"]
            WidthFno = Config["general"]["WidthFno"]
            path_model_Fno = snapshot_path+ f'/Fno_{ModesFno}_{WidthFno}_SepFNODeCoder{Config["general"]["SepFNODeCoder"]}_{epoch_num}.pth'
            
            #load the model
            checkpoint = torch.load(path_model_Fno)
            if Config["general"]["SepFNODeCoder"] == "Yes":
                model.model_v.load_state_dict(checkpoint['model_v'])
                model.remain_model_v.load_state_dict(checkpoint['remain_model_v'])
            elif Config["general"]["SepFNODeCoder"] == "No":
                model.model_v.load_state_dict(checkpoint['model_v'])
       

    ### Load trained model
    # /home/nellie/data/project_complex/OASIS32D_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.03-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Register_19999.pth 
    # /home/nellie/data/project_complex/OASIS32D_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.03-1.0_EPDiff-Mse-Relative-100-100_nSteps7/ATT_19999.pth 
    # /home/nellie/data/project_complex/OASIS32D_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.03-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Fno_8_20_SepFNODeCoderYes_19999.pth
    path_model_Register = "/home/nellie/data/project_complex/OASIS32D_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.03-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Register_19999.pth"
    path_model_ATT = "/home/nellie/data/project_complex/OASIS32D_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.03-1.0_EPDiff-Mse-Relative-100-100_nSteps7/ATT_19999.pth"
    path_model_Fno = "/home/nellie/data/project_complex/OASIS32D_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.03-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Fno_8_20_SepFNODeCoderYes_19999.pth"



    #/home/nellie/data/project_complex/PlantGray4_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.01-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Register_19999.pth 
    #/home/nellie/data/project_complex/PlantGray4_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.01-1.0_EPDiff-Mse-Relative-100-100_nSteps7/ATT_19999.pth 
    # /home/nellie/data/project_complex/PlantGray4_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.01-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Fno_8_20_SepFNODeCoderYes_19999.pth
    path_model_Register = "/home/nellie/data/project_complex/PlantGray4_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.01-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Register_19999.pth"
    path_model_ATT = "/home/nellie/data/project_complex/PlantGray4_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.01-1.0_EPDiff-Mse-Relative-100-100_nSteps7/ATT_19999.pth"
    path_model_Fno = "/home/nellie/data/project_complex/PlantGray4_128/JulyGDN_lr0.0005/SATT101_UPS2Ep20000_ATTblocks1_bs12_lrR0.0005_lrE0_lddmm-alpha-gamma-power-1.0-0.5-2.0_loss-simi-sigma-reg-0.5-0.01-1.0_EPDiff-Mse-Relative-100-100_nSteps7/Fno_8_20_SepFNODeCoderYes_19999.pth"
    
    
    #load the model
    checkpoint = torch.load(path_model_Register)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.remain.load_state_dict(checkpoint['remain'])
    # model.SelfAttention.load_state_dict(checkpoint['SelfAttention'])
    if hasattr(model, 'SelfAttention'):
        assert 356>444
        checkpoint = torch.load(path_model_ATT)
        model.SelfAttention.load_state_dict(checkpoint['SelfAttention'])
    #load the model
    checkpoint = torch.load(path_model_Fno)
    if Config["general"]["SepFNODeCoder"] == "Yes":
        model.model_v.load_state_dict(checkpoint['model_v'])
        model.remain_model_v.load_state_dict(checkpoint['remain_model_v'])
    elif Config["general"]["SepFNODeCoder"] == "No":
        model.model_v.load_state_dict(checkpoint['model_v'])
    

    test_loss = {"total_loss":[], "loss_regis":[], "similarity":[], "regularity":[], "Fno": [], "Fno_mse": [], "dice": []}
    test_result = {"src":[], "tar":[], "Sdef":[], "phiinv":[], "phi":[], "velocity":[], "VList":[], "VList_gt":[], "Phiinv_gt":[], "Sdef_gt":[], "Dispinv":[]}  
    test_result_V_M = {"VList":[], "VList_gt":[], "MList":[], "MList_gt":[], 'Reg_GDN':[], 'Reg_EPDiff':[]} 

    # tslen = 320
    tslen = 100
    
    # testloader = loadTest(Config,bs=1,tslen=20)  #for registration
    if Config["DataSet"]["dataName"] in ["OASIS32D"] or "Plant" in Config["DataSet"]["dataName"]:
        testloader, text_data = loadTest(Config,bs=1,tslen=tslen)
    else:
        testloader = loadTest(Config,bs=1,tslen=tslen)


    
    ''' def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    get_parameter_number(model) '''

    def dumpModelSize(model, details=True):
        total = sum(p.numel() for p in model.parameters())
        if details:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_params = sum(p.numel() for p in param)
                    print(f"name: {name}, num params: {num_params} ({(num_params/total) *100 :.2f}%)")

        print(f"total params: {total}, ", end='')
        print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    dumpModelSize(model)
    # assert 3>333
    


    src_list = []; tar_list = []; velocity_list = []; sdef_list = []; text_Id_list = []
    for i_batch, sampled_batch in enumerate(testloader): 
        ''' if i_batch !=9:
            continue
        else:
            srcRGB = sampled_batch['srcRGB'][0].cpu().detach().numpy()
            tarRGB = sampled_batch['tarRGB'][0].cpu().detach().numpy() '''
        
        
            # print(srcRGB.dtype)
            # save as png
            # print(srcRGB.shape, tarRGB.shape, np.max(srcRGB), np.min(srcRGB), np.max(tarRGB), np.min(tarRGB))
            # assert 3>333
            #save srcRGB and tarRGB as numpy 
            # np.save('/home/nellie/code/MeDIT-master_nii_show/ICLR2024/srcRGB.npy', srcRGB)
            # np.save('/home/nellie/code/MeDIT-master_nii_show/ICLR2024/tarRGB.npy', tarRGB)
            # assert 1>222

        if Config["DataSet"]["dataName"] in ["Plant"]:
            srcRGB = sampled_batch['srcRGB']
            tarRGB = sampled_batch['tarRGB']
            src = sampled_batch['src']; tar = sampled_batch['tar']
            b,b,c,w,h = srcRGB.shape
            src = srcRGB.reshape(-1,c,w,h).cuda(); tar = tarRGB.reshape(-1,c,w,h).cuda()  #[10, 3, 128, 128]
        elif Config["DataSet"]["dataName"] in ["PlantR2"]:
            srcRGB = sampled_batch['srcRGB'][:,:,-1:,:,:]
            tarRGB = sampled_batch['tarRGB'][:,:,-1:,:,:]
            b,b,c,w,h = srcRGB.shape
            src = srcRGB.reshape(-1,c,w,h).cuda(); tar = tarRGB.reshape(-1,c,w,h).cuda()  #[10, 3, 128, 128]
        else:  #["PlantGray", "OASIS32D", "Mnist", "BullEye", "Shape", "PlantGray2"]
            # print(i_batch)
            src, tar = sampled_batch['src'], sampled_batch['tar']
            b,b,w,h = src.shape
            src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]

            if "srcRGB" in sampled_batch.keys():
                srcRGB = sampled_batch['srcRGB'].cuda()
                tarRGB = sampled_batch['tarRGB'].cuda()
            elif "textidx" in sampled_batch.keys():
                textid = sampled_batch['textidx']
                srcRGB = None
                tarRGB = None
            else:
                srcRGB = None
                tarRGB = None


            # #save src and tar as numpy
            # np.save('/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/Img/src.npy', src.cpu().detach().numpy())
            # np.save('/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Mnist/Img/tar.npy', tar.cpu().detach().numpy()) 
        # np.save('/home/nellie/data/DealData/src.npy', src.cpu().detach().numpy())
        # np.save('/home/nellie/data/DealData/tar.npy', tar.cpu().detach().numpy())
        
        #zoom it to 256*256
        # src = zoom(src.cpu().detach().numpy(), 0.5, order=1); src = torch.from_numpy(src).float().cuda().unsqueeze(1)
        # tar = zoom(tar.cpu().detach().numpy(), 0.5, order=1); tar = torch.from_numpy(tar).float().cuda().unsqueeze(1)
        
        # src = src[:,::2,::2]; tar = tar[:,::2,::2]

        
        # v_seq_full_dim, v_seq_gt_epdiff, Sdef, Sdfm_binary, m_seq_full_dim, phiinv_pred = model(torch.cat((src,tar),dim=1))
        inputS = torch.cat((src,tar),dim=1)
        

        # SdefList, phiinvList, phiList, velocity= model(inputS)
        """ with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            SdefList, phiinvList, phiList, V_List, V_List_gt, M_List = model(inputS)
            velocity = V_List[0]
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        assert 3>333 """
        # Sdef_List, Phiinv_List, Phi_List, V_List, V_List_gt, M_List
        

        if srcRGB is not None:
            SdefList, phiinvList, phiList, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt, srcRGB_List, srcRGB_List_gt = model(inputS, altarFlag=altarFlag, srcRGB = srcRGB)
        else:
            SdefList, phiinvList, phiList, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt = model(inputS, altarFlag=altarFlag)
        
        




        ### plants
        srcRGB_List = torch.stack(srcRGB_List, dim=1)
        srcRGB_List_gt = torch.stack(srcRGB_List_gt, dim=1)

        ### OASIS32D_128
        #save all SdefList, phiinvList, phiList, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt as npy
        # SdefList = torch.stack(SdefList, dim=1) #[410, 7, 1, 128, 128]
        # Sdef_List_gt = torch.stack(Sdef_List_gt, dim=1) #[410, 7, 1, 128, 128]

        V_List = torch.stack(V_List, dim=1) #[410, 2, 128, 128]
        V_List_gt = torch.stack(V_List_gt, dim=1) #[410, 2, 128, 128]

        phiList = torch.stack(phiList, dim=1) #[410, 2, 128, 128]
        Phi_List_gt = torch.stack(Phi_List_gt, dim=1) #[410, 2, 128, 128]

        phiinvList = torch.stack(phiinvList, dim=1) #[410, 2, 128, 128]
        Phiinv_List_gt = torch.stack(Phiinv_List_gt, dim=1) #[410, 2, 128, 128]

        # torch.Size([16, 7, 1, 128, 128]) torch.Size([16, 7, 2, 128, 128]) torch.Size([16, 7, 2, 128, 128]) torch.Size([16, 7, 2, 128, 128])
        # print(src.shape, tar.shape) #torch.Size([16, 1, 128, 128]) torch.Size([16, 1, 128, 128])
        # print(SdefList.shape, V_List.shape, phiList.shape, phiinvList.shape)
        # print(Sdef_List_gt.shape, V_List_gt.shape, Phi_List_gt.shape, Phiinv_List_gt.shape)

        #torch.Size([62, 7, 1, 128, 128, 3]) torch.Size([62, 7, 2, 128, 128]) torch.Size([62, 7, 2, 128, 128]) torch.Size([62, 7, 2, 128, 128])
        print(src.shape, tar.shape) #torch.Size([16, 1, 128, 128]) torch.Size([16, 1, 128, 128])
        print(srcRGB_List.shape, V_List.shape, phiList.shape, phiinvList.shape)
        print(srcRGB_List_gt.shape, V_List_gt.shape, Phi_List_gt.shape, Phiinv_List_gt.shape)
        #src tar SdefList V_List phiList phiinvList V_List_gt Phi_List_gt Phiinv_List_gt Sdef_List_gt
        #save as npy
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-SdefList.npy', srcRGB_List.cpu().detach().numpy())  
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-V_List.npy', V_List.cpu().detach().numpy())
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-phiList.npy', phiList.cpu().detach().numpy())
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-phiinvList.npy', phiinvList.cpu().detach().numpy())
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-V_List_gt.npy', V_List_gt.cpu().detach().numpy())
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-Phi_List_gt.npy', Phi_List_gt.cpu().detach().numpy())
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-Phiinv_List_gt.npy', Phiinv_List_gt.cpu().detach().numpy())
        np.save('/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/AE/Plant-Sdef_List_gt.npy', srcRGB_List_gt.cpu().detach().numpy())
        assert 3>333
        
        



        






        # all_velocity = torch.cat(V_List, dim=1)
        # save all_velocity_as npy
        # np.save(f'/home/nellie/data/project_complex/PlantGray4_128/JulyGDN_lr0.0005/velocity_groundth_{V_List[0].shape[0]}.npy', all_velocity.detach().cpu().numpy())
        # np.save(f'/home/nellie/data/project_complex/OASIS32D_128/JulyGDN_lr0.0005/velocity_groundth_{V_List[0].shape[0]}.npy', all_velocity.detach().cpu().numpy())
        # print(len(V_List), V_List[0].shape)
        # assert 2>333
        
        
        # torch.Size([410, 1, 128, 128]) torch.Size([410, 1, 128, 128]) (1, 204, 204, 3) (1, 204, 204, 3) uint8 uint8
        # print(src.shape, tar.shape, srcRGB.shape, tarRGB.shape, srcRGB.dtype, tarRGB.dtype, sampled_batch['src_time'].shape, sampled_batch['tar_time'].shape)
        # assert 1>222
        
        # print(SdefList[-1].shape)
        # assert 2>333
        velocity = V_List[0] #[320, 2, 128, 128]
        Sdef = SdefList[-1]
        # print(Sdef.shape, tar.shape)  #RGB: ([1, 3, 128, 128])   ([1, 3, 128, 128])
        # print(tar.shape, Sdef.shape, velocity.shape, len(SdefList), len(phiinvList), len(phiList), len(velocity))
        # assert 3>333
        loss1 = criterion(tar,Sdef)
        loss2 = regloss(None, velocity)
        loss = loss1*WSimi + WReg*loss2


        SdefList = torch.stack(SdefList, dim=1) #[410, 7, 1, 128, 128]
        

        Data_for_Nivetha = False
        if Data_for_Nivetha == True:
            if srcRGB is not None:
                SRGBdefList = torch.stack(srcRGB_List, dim=1) #[410, 7, 3, 128, 128]
                print(SRGBdefList.shape, SRGBdefList.dtype)
                traindata = {
                    "src": src.cpu().detach().numpy(),
                    "tar": tar.cpu().detach().numpy(),
                    "srcRGB": srcRGB.cpu().detach().numpy(),
                    "tarRGB": tarRGB.cpu().detach().numpy(),
                    "src_time": sampled_batch['src_time'].cpu().detach().numpy(),
                    "tar_time": sampled_batch['tar_time'].cpu().detach().numpy(),
                    "SdefList": SdefList.cpu().detach().numpy(),
                    "SRGBdefList": SRGBdefList.cpu().detach().numpy()
                }
                #save as .mat
                import scipy.io as sio
                sio.savemat('/home/nellie/data/DealData/For_Nivetha/plants_testdata.mat', traindata)
                print(len(SdefList), SdefList[0].shape)
                assert 2>333
            else:
            

                # src_list = []; tar_list = []; velocity_list = []; sdef_list = []; text_Id_list = []
                src_list.append(src.cpu().detach().numpy())
                tar_list.append(tar.cpu().detach().numpy())
                sdef_list.append(SdefList.cpu().detach().numpy())
                text_Id_list.append(textid.cpu().detach().numpy())


            # if i_batch < 3:
            #     continue;
            # elif i_batch >= 3:
            src = np.concatenate(src_list, axis=0)
            tar = np.concatenate(tar_list, axis=0)
            SdefList = np.concatenate(sdef_list, axis=0)
            textid = np.concatenate(text_Id_list, axis=0)
            print(src.shape, tar.shape, SdefList.shape, textid.shape)

            
            traindata = {
                "src": src,
                "tar": tar,
                "textid": textid,
                "SdefList": SdefList,
                "text_data": text_data
            }

            import scipy.io as sio
            sio.savemat('/home/nellie/data/DealData/For_Nivetha/OASIS32D_testdata.mat', traindata)
            print(len(SdefList), SdefList[0].shape)
            assert 3>444

        
        
        


        # add another channel to velocity with the values equal to zero, the final shape is [320, 3, 128, 128]
        """ velocity_save = torch.cat((velocity, torch.zeros_like(velocity[:,0:1,:,:])), dim=1)
        velocity_save = velocity_save.permute(0,2,3,1).cpu().detach().numpy()
        #save as nii.gz
        velocity_save = velocity_save[0:1]
        sitk.WriteImage(sitk.GetImageFromArray(velocity_save), f"/p/mmcardiac/nellie/data/project_complex/PlantGray_128/v0_{i_batch}.nii.gz")
        velocity_last = V_List[-1]
        velocity_last = torch.cat((velocity_last, torch.zeros_like(velocity_last[:,0:1,:,:])), dim=1)
        velocity_last = velocity_last.permute(0,2,3,1).cpu().detach().numpy()
        #save as nii.gz
        velocity_last = velocity_last[0:1]
        sitk.WriteImage(sitk.GetImageFromArray(velocity_last), f"/p/mmcardiac/nellie/data/project_complex/PlantGray_128/v_last_{i_batch}.nii.gz")
        assert 3>333 """


        # for i in range(1, len(V_List)):
        #     print(i, torch.max(V_List_gt[i]).item(), torch.min(V_List_gt[i]).item(), torch.mean(V_List_gt[i]).item())
        #     print(i, torch.max(V_List[i]).item(), torch.min(V_List[i]).item(), torch.mean(V_List[i]).item())
        #     print("\n")
        # # np.save('/home/nellie/code/cvpr/ComplexNet/TTTTMid-Res/Shape/velocity_Vm_1000.0.npy', velocity.detach().cpu().numpy())
        # assert 3>222

        #M_List V_List   M_list_gt V_List_gt
        ''' Reg_GDN = [V.detach().cpu()*M.detach().cpu() for V,M in zip(V_List,M_List)] 
        Reg_EPDiff = [V.detach().cpu()*M.detach().cpu() for V,M in zip(V_List_gt,M_list_gt)]
        test_result_V_M['Reg_GDN'].append(Reg_GDN)
        test_result_V_M['Reg_EPDiff'].append(Reg_EPDiff) '''

        
        


        

        print(f"loss1 {loss1.item()} loss2 {loss2.item()} loss {loss.item()}")

        test_loss['total_loss'].append(loss.item())
        test_loss['similarity'].append(loss1.item())
        test_loss['regularity'].append(loss2.item())
        print(src.shape,         tar.shape,            SdefList[0].shape,       phiinvList[0].shape,    phiList[0].shape,       velocity.shape, len(SdefList), len(phiinvList), len(phiList), len(velocity))
        # # [320, 1, 128, 128])   [320, 1, 128, 128])  [320, 1, 128, 128]      ([320, 128, 128, 2])    ([320, 128, 128, 2])    ([320, 2, 128, 128]) 7 7 7 320
        # [10, 1, 128, 128])     [10, 1, 128, 128])      [10, 1, 128, 128])     [10, 2, 128, 128])       [10, 2, 128, 128])     [10, 2, 128, 128]) 7 7 7 10
        # assert 3>333
        if Config["DataSet"]["dataName"] in ["Plant"]: #RGB: ([1, 3, 128, 128])   ([1, 3, 128, 128])
            Sdef = RGB2GRAY(Sdef)
            tar = RGB2GRAY(tar)
            src = RGB2GRAY(src)
            SdefList = [RGB2GRAY(Sdef) for Sdef in SdefList]
            Sdef_List_gt = [RGB2GRAY(Sdef) for Sdef in Sdef_List_gt]

            



        test_result['src'].append(src)
        test_result['tar'].append(tar)
        test_result['Sdef'].append(SdefList)
        test_result['Sdef_gt'].append(Sdef_List_gt)
        test_result['phiinv'].append(phiinvList)
        test_result['phi'].append(phiList)
        test_result['velocity'].append(V_List)
        test_result['VList_gt'].append(V_List_gt)
        test_result['Phiinv_gt'].append(Phiinv_List_gt)
        test_result['Dispinv'].append(Dispinv_List)
        


        src_list.append(src.cpu().detach().numpy())
        tar_list.append(tar.cpu().detach().numpy())
        velocity_list.append(velocity.cpu().detach().numpy())
    
    src_list = np.concatenate(src_list, axis=0)
    tar_list = np.concatenate(tar_list, axis=0)
    velocity_list = np.concatenate(velocity_list, axis=0)
    print(src_list.shape, tar_list.shape, velocity_list.shape)

    ''' Reg_GDN_list = test_result_V_M['Reg_GDN']  #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]
    Reg_EPDiff_list = test_result_V_M['Reg_EPDiff'] #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]
    step_GDN_mean_all_list = []; step_EPDiff_mean_all_list = []
    for step in range(7):
        step_GDN = [item[step] for item in Reg_GDN_list] #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 1
        step_EPDiff = [item[step] for item in Reg_EPDiff_list] #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]

        step_GDN = torch.cat(step_GDN, dim=0) #[10*16, 2, 128, 128]
        step_EPDiff = torch.cat(step_EPDiff, dim=0) #[10*16, 2, 128, 128]

        step_GDN_mean = torch.mean(step_GDN, dim=(1,2,3)) #`torch.Size([160])`
        step_EPDiff_mean = torch.mean(step_EPDiff, dim=(1,2,3)) #`torch.Size([160])`

        step_GDN_mean_all_list.append(step_GDN_mean)
        step_EPDiff_mean_all_list.append(step_EPDiff_mean)


    step_GDN_mean_all = torch.stack(step_GDN_mean_all_list, dim=1) #[160, 7]
    step_EPDiff_mean_all = torch.stack(step_EPDiff_mean_all_list, dim=1) #[160, 7]

    #write the [160, 7] to the txt one line by one line
    for i in range(step_GDN_mean_all.shape[0]):
        step_GDN_mean = step_GDN_mean_all[i]
        step_EPDiff_mean = step_EPDiff_mean_all[i]
        textf_GDN_DiFuS.write(' '.join([str(item) for item in step_GDN_mean.tolist()]) + '\n')
        textf_EPDiff.write(' '.join([str(item) for item in step_EPDiff_mean.tolist()]) + '\n')
    textf_GDN_DiFuS.close()
    textf_EPDiff.close() '''


    # assert 3>333
    # save them as .mat
    # import scipy.io as sio
    # sio.savemat('/home/nellie/data/src_tar_v0_5000pairs.mat', {'src':src_list, 'tar':tar_list, 'velocity':velocity_list})
    # assert 1>333
    if Config["DataSet"]["test_type"] == 'Mnist':
        draw_one_picture(test_result, Config, Snum=10, TotalNum=4)
    elif Config["DataSet"]["test_type"] == 'Shape' or Config["DataSet"]["test_type"] == 'BullEye':
        draw_one_picture(test_result, Config)
    elif 'Plant' in Config["DataSet"]["test_type"] or Config["DataSet"]["test_type"] == 'OASIS32D':
        # draw_one_picture2(test_result, Config, Snum=1, TotalNum=3)
        draw_one_picture2(test_result, Config, Snum=3, TotalNum=tslen, num_per_picture=3*5)


    logging.info(f"VoxelmorphTest~~~~~~ totalloss {np.mean(test_loss['total_loss'])} similarity {np.mean(test_loss['similarity'])} regularity {np.mean(test_loss['regularity'])}")

def tester_GDN_DiFuS(Config, model):
    SRGBdefList_TEST = None
    visjpgpath = Config["general"]["visjpg"]
    random.seed(Config["general"]["seed"])
    np.random.seed(Config["general"]["seed"])
    torch.manual_seed(Config["general"]["seed"])
    torch.cuda.manual_seed(Config["general"]["seed"])

    WSimi = Config["LossWeightSvf"]["WSimi"]
    WReg = Config["LossWeightSvf"]["WReg"]
    snapshot_path = Config["general"]["snapshot_path"]
    # cond_scale=1.5
    # src_cond_scale=2.0

    cond_scale = Config["general"]["cond_scale"]
    src_cond_scale = Config["general"]["src_cond_scale"]

    test_loss = {"total_loss":[], "loss_regis":[], "similarity":[], "regularity":[], "Fno": [], "Fno_mse": [], "dice": []}
    test_result = {"src":[], "tar":[], "Sdef":[], "phiinv":[], "phi":[], "velocity":[], "VList":[], "VList_gt":[], "Phiinv_gt":[], "Sdef_gt":[], "Dispinv":[], "MList":[], "MList_gt":[]}  
    test_result_V_M = {"VList":[], "VList_gt":[], "MList":[], "MList_gt":[], 'Reg_GDN':[], 'Reg_EPDiff':[]}
    test_result_DiFuS = {"FVD": [], "FID": [], "KID": [], "IS": []} 

    # testloader = loadTest(Config,bs=1,tslen=20)  #for registration
    fixed_src = Config["general"]["fixed_src"]
    Shape_Tag = Config["DataSet"]["Shape_Tag"]
    Mnist_Tag = Config["DataSet"]["Mnist_Tag"]
    use_bert_text_cond = False

    if fixed_src == -1: #Don’t fix src=-1   计算velocity distribution and regularity
        tslen = 40
        if Config["DataSet"]["test_type"] == 'Mnist':
            testloader = loadTest(Config,bs=1,tslen=tslen,testnumMnist=[Mnist_Tag])
            TestDateStr = f"Mnist-{Mnist_Tag}"
        elif Config["DataSet"]["test_type"] == 'Shape':
            testloader = loadTest(Config,bs=1,tslen=tslen,testnumShape=[Shape_Tag])
            TestDateStr = f"Shape-{Shape_Tag}"
        else: #BullEye Plant OASIS32D
            TestDateStr = f'{Config["DataSet"]["test_type"]}'
            if Config["DataSet"]["test_type"] == 'BullEye':
                testloader = loadTest(Config,bs=1,tslen=tslen)
            elif Config["DataSet"]["dataName"] in ["OASIS32D"] or "Plant" in Config["DataSet"]["dataName"]:
                # testloader, text_data = loadTest(Config,bs=16,tslen=160)
                # testloader, text_data = loadTest(Config,bs=4,tslen=40)
                testloader, text_data = loadTest(Config,bs=16,split="validate-on-test")   #On training set, use tslen=40
                use_bert_text_cond = True
                # use_bert_text_cond = False

        textpath = f"{visjpgpath}/text"
        textf_GDN_DiFuS_path = f"{textpath}/reg_{TestDateStr}_GDN_DiFuS.txt"
        textf_EPDiff_path = f"{textpath}/reg_{TestDateStr}_EPDiff.txt"
        textf_text_cond_path = f"{textpath}/UnFixSrc_{TestDateStr}_text_cond.txt"
        if not os.path.exists(textpath):
            os.makedirs(textpath)
        textf_GDN_DiFuS = open(textf_GDN_DiFuS_path, 'w')
        textf_EPDiff = open(textf_EPDiff_path, 'w')
        textf_text_cond = open(textf_text_cond_path, 'w')

    else: #compute the confidence map
        TestDateStr = f'{Config["DataSet"]["test_type"]}'
        testloader, text_data = loadTest(Config, fixed_src=fixed_src)   #On training set, use tslen=40
        use_bert_text_cond = True

        textpath = f"{visjpgpath}/text"
        textf_text_cond_path = f"{textpath}/FixSrc_{fixed_src}_{TestDateStr}_text_cond.txt"
        if not os.path.exists(textpath):
            os.makedirs(textpath)
        textf_text_cond = open(textf_text_cond_path, 'w')
    

    from networks.video_diffusion_pytorch import Trainer_with_GDN, Unet3D, GaussianDiffusion, Trainer
    
    diffusion = GaussianDiffusion(
        Unet3D(
            dim = 32,
            dim_mults = (1, 2, 4),
            channels = 20+Config["general"]["ConditionCH"],
            out_dim = 20,
            use_bert_text_cond = use_bert_text_cond
        ),
        image_size = 16,
        num_frames = 7,
        channels = 20+Config["general"]["ConditionCH"],
        timesteps = 500,   # number of steps
        loss_type = 'l2',    # L1 or L2
        out_dim = 20
    ).cuda()
        

    diffusion_trainer = Trainer_with_GDN(
        diffusion,
        # train_lr = 1e-4,
        # train_lr = LR,
        save_and_sample_every = 1000,
        # save_and_sample_every = 1,
        train_num_steps = 100000,         # total training steps
        # gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                        # turn on mixed precision
        results_folder = snapshot_path,
    )

    visjpg = Config["general"]["visjpg"]
    
    # diffusion_trainer.load_model2(Config)
    #load pre-trained model of Register
    LR = Config["LStrategy"]["Registration"]["lr"]
    snapshot_path_ = snapshot_path.split('_way')[0]
    snapshot_path_ = snapshot_path_.replace(f'JulyGDN_DifuS_lr{LR}', 'JulyGDN_lr0.0005').replace(f'lrR{LR}', 'lrR0.0005').replace(f'bs{Config["LStrategy"]["train_batch_size"]}', 'bs12')
    path_model_Register = snapshot_path_+ "/Register_{}.pth".format(Config["LStrategy"]["max_epochs"] - 1)
    ModesFno = Config["general"]["ModesFno"]
    WidthFno = Config["general"]["WidthFno"]
    path_model_Fno = snapshot_path_+ f'/Fno_{ModesFno}_{WidthFno}_SepFNODeCoder{Config["general"]["SepFNODeCoder"]}_{Config["LStrategy"]["max_epochs"] - 1}.pth'

    #load the model
    checkpoint = torch.load(path_model_Register)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.remain.load_state_dict(checkpoint['remain'])
    # model.SelfAttention.load_state_dict(checkpoint['SelfAttention'])
    print(hasattr(model, 'SelfAttention'), Config["general"]["SepFNODeCoder"])  #False Yes
    print("\n\n", path_model_Register, "\n\n", "\n\n", path_model_Fno, "\n\n")
    #load the model
    checkpoint = torch.load(path_model_Fno)
    model.model_v.load_state_dict(checkpoint['model_v'])
    model.remain_model_v.load_state_dict(checkpoint['remain_model_v'])
    model.freeze_model_v()
    model.freeze_encoder_and_attention()

    sampled_videos_list = []; V_Z_List_NotUse_list = [];src_cond_list = []

    src_list = []; tar_list = []; velocity_list = []; text_cond_list = []

    all_sample_vs_list_list = []; all_gtruth_vs_list_list = []; all_sample_dfm_list_list = []; 
    all_sample_dfm_gt_list_list = []; all_testdataset_dfm_list_list = []
    all_sample_dfm_dec_list_list = []
    all_dec_vs_list = []; all_dec_vs_list_list = []
    all_dec_phi_list = []; all_dec_phi_list_list = []
    all_dec_phiinv_list = []; all_dec_phiinv_list_list = []
    


    for i_batch, sampled_batch in enumerate(testloader): 
        text_cond = None
        src, tar = sampled_batch['src'], sampled_batch['tar']
        b,b,w,h = src.shape
        src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]

        if "srcRGB" in sampled_batch:
            srcRGB = sampled_batch['srcRGB'].cuda()
            tarRGB = sampled_batch['tarRGB'].cuda()
            SRGBdefList_TEST = sampled_batch['SRGBdefList'].cuda()
        else:
            SdefList_TEST = sampled_batch['SdefList'].cuda()

        if 'textidx' in sampled_batch:
            textidx = sampled_batch['textidx']
            textidx = textidx.reshape(-1)
            text_cond = get_text_list(textidx, text_data)
            for text in text_cond:
                text_cond_list.append(text)

        if "src_time" in sampled_batch:
            src_time = sampled_batch['src_time']; src_time = src_time.reshape(-1)
            tar_time = sampled_batch['tar_time']; tar_time = tar_time.reshape(-1)
            text_cond = get_text_list_from_time(src_time, tar_time)
            for text in text_cond:
                text_cond_list.append(text)
    
        if src.shape[0] < 16:
            src = torch.concat([src, src[:16%src.shape[0]]], dim=0)
            tar = torch.concat([tar, tar[:16%tar.shape[0]]], dim=0)

            if "srcRGB" in sampled_batch:
                srcRGB = torch.concat([srcRGB, srcRGB[:16%srcRGB.shape[0]]], dim=0)
                tarRGB = torch.concat([tarRGB, tarRGB[:16%tarRGB.shape[0]]], dim=0)
                SRGBdefList_TEST = torch.concat([SRGBdefList_TEST, SRGBdefList_TEST[:16%SRGBdefList_TEST.shape[0]]], dim=0)
                src_time = torch.concat([src_time, src_time[:16%len(src_time)]], dim=0)
                tar_time = torch.concat([tar_time, tar_time[:16%len(tar_time)]], dim=0)
            else:
                SdefList_TEST = torch.concat([SdefList_TEST, SdefList_TEST[:16%SdefList_TEST.shape[0]]], dim=0)
            text_cond = text_cond + text_cond[:16%len(text_cond)]
        
        if SRGBdefList_TEST is not None:
            SRGBdefList_TEST = [SRGBdefList_TEST[:,idx] for idx in range(SRGBdefList_TEST.shape[1])]
            SdefList_TEST = None
        else:
            SdefList_TEST = [SdefList_TEST[:,idx] for idx in range(SdefList_TEST.shape[1])]
            SRGBdefList_TEST = None
        
        if "Plant" in Config["DataSet"]["dataName"]:
            src_cond = get_source_gradient(src) * 5
        elif Config["DataSet"]["dataName"] in ["Shape"]:
            src_cond = get_source_gradient(src) * 2
        else:
            src_cond = get_source_gradient(src) * 1

        src_cond = src_cond.unsqueeze(2)[:,:,:,::8,::8]
        src_cond = src_cond.repeat(1,1,7,1,1)
        i_batch_str = f"{Config['DataSet']['test_type']}_{i_batch}"

        if fixed_src != -1:
            i_batch_str = f"FixSrc_{fixed_src}_{i_batch_str}"
        else:
            i_batch_str = f"UnFixSrc_{i_batch_str}"
        
        sampled_videos = diffusion_trainer.model.sample(src_cond = src_cond, cond = text_cond, cond_scale=cond_scale, src_cond_scale=src_cond_scale)
        print("sampled_videos: ",sampled_videos.shape) #[16, 7, 20, 16, 16]


        sampled_videos_list.append(sampled_videos.detach().cpu().numpy())
        src_cond_list.append(src_cond.detach().cpu().numpy())
        V_Z_List = torch.unbind(sampled_videos, dim=2)


        #Sdef_List Sdef_List_gt   srcRGB_List srcRGB_List_gt #Nellie
        if "srcRGB" in sampled_batch:
            SdefList, phiinvList, phiList, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt, \
                V_Z_List_Encoder, srcRGB_List, srcRGB_List_gt,\
                V_List_Dec, Sdef_List_Dec, srcRGB_List_Dec = model(V_Z_List,src,torch.cat((src,tar),dim=1),srcRGB=srcRGB)
        else:
            SdefList, phiinvList, phiList, V_List, V_List_gt, M_List, Phiinv_List_gt, Sdef_List_gt, Dispinv_List, Phi_List_gt, M_list_gt, \
                V_Z_List_Encoder,\
                V_List_Dec, Sdef_List_Dec = model(V_Z_List,src,torch.cat((src,tar),dim=1),srcRGB=None)

    
        

        # assert 3>444
        # 计算一些quantitative metrics
        velocity = V_List[0]
        V_Z_List_NotUse_list.append(V_Z_List_Encoder.detach().cpu().numpy())



        if fixed_src == -1: #regularity and v distribution
            #计算trajectories of regularity, sampled 和 epdiff,     velocity
            #M_List V_List      M_list_gt V_List_gt
            Reg_GDN = [V.detach().cpu()*M.detach().cpu() for V,M in zip(V_List,M_List)] 
            Reg_EPDiff = [V.detach().cpu()*M.detach().cpu() for V,M in zip(V_List_gt,M_list_gt)]
            test_result_V_M['Reg_GDN'].append(Reg_GDN)
            test_result_V_M['Reg_EPDiff'].append(Reg_EPDiff)

            #计算FVD, FID KID IS, srcRGB_List 和 srcRGB_List_gt
            # FID KID
            # print("Start Computing:   ", len(SRGBdefList_TEST), len(srcRGB_List), SRGBdefList_TEST[0].shape, srcRGB_List[0].shape)  #7 7   ([16, 1, 128, 128, 3])   ([16, 1, 128, 128, 3])
            # assert 2>333
            if "srcRGB" in sampled_batch:
                KID_Value = [KID(realImg=srcRGB.cpu(), sampledImg=srcRGB_gt.cpu()) for (srcRGB, srcRGB_gt) in zip(srcRGB_List, SRGBdefList_TEST)]
                FID_Value = [FID(realImg=srcRGB.cpu(), sampledImg=srcRGB_gt.cpu()) for (srcRGB, srcRGB_gt) in zip(srcRGB_List, SRGBdefList_TEST)]
                test_result_DiFuS['KID'].append(KID_Value)
                test_result_DiFuS['FID'].append(FID_Value)

                print(f"KID_Value:\n {KID_Value}")
                print(f"FID_Value:\n {FID_Value}")
                
            else:
                # print(SdefList_TEST[0].shape)  #7, 1, 128, 128]
                # assert 231>333
                pass
                ''' KID_Value = [KID(realImg=src.cpu(), sampledImg=src_gt.cpu()) for (src, src_gt) in zip(SdefList, SdefList_TEST)]
                FID_Value = [FID(realImg=src.cpu(), sampledImg=src_gt.cpu()) for (src, src_gt) in zip(SdefList, SdefList_TEST)]
                test_result_DiFuS['KID'].append(KID_Value)
                test_result_DiFuS['FID'].append(FID_Value)
                print(f"KID_Value:\n {KID_Value}")
                print(f"FID_Value:\n {FID_Value}") '''


            Reg_GDN_list = test_result_V_M['Reg_GDN']  #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]
            Reg_EPDiff_list = test_result_V_M['Reg_EPDiff'] #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]
            step_GDN_mean_all_list = []; step_EPDiff_mean_all_list = []
            for step in range(7):
                step_GDN = [item[step] for item in Reg_GDN_list] #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 1
                step_EPDiff = [item[step] for item in Reg_EPDiff_list] #[16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]

                step_GDN = torch.cat(step_GDN, dim=0) #[10*16, 2, 128, 128]
                step_EPDiff = torch.cat(step_EPDiff, dim=0) #[10*16, 2, 128, 128]

                step_GDN_mean = torch.mean(step_GDN, dim=(1,2,3)) #`torch.Size([160])`
                step_EPDiff_mean = torch.mean(step_EPDiff, dim=(1,2,3)) #`torch.Size([160])`

                step_GDN_mean_all_list.append(step_GDN_mean)
                step_EPDiff_mean_all_list.append(step_EPDiff_mean)

            step_GDN_mean_all = torch.stack(step_GDN_mean_all_list, dim=1) #[160, 7]
            step_EPDiff_mean_all = torch.stack(step_EPDiff_mean_all_list, dim=1) #[160, 7]

            #write the [160, 7] to the txt one line by one line
            for i in range(step_GDN_mean_all.shape[0]):
                step_GDN_mean = step_GDN_mean_all[i]
                step_EPDiff_mean = step_EPDiff_mean_all[i]
                textf_GDN_DiFuS.write(' '.join([str(item) for item in step_GDN_mean.tolist()]) + '\n')
                textf_EPDiff.write(' '.join([str(item) for item in step_EPDiff_mean.tolist()]) + '\n')
            

        


        

        # test_result_DiFuS = {"FVD": [], "FID": [], "KID": [], "IS": []} 
        # [62, 1, 128, 128, 3] [62, 1, 128, 128, 3]  [62, 1, 128, 128, 3]
        # print(srcRGB.shape, srcRGB_List[0].shape, srcRGB_List_gt[0].shape, srcRGB_List[0].dtype, torch.max(srcRGB_List[0]), torch.min(srcRGB_List[0]))
        # assert 1>222

        # imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        # imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        # fid.update(imgs_dist1, real=True)
        # fid.update(imgs_dist2, real=False)
        # fid.compute()




        # V_List  [ [16, 2, 128, 128] [16, 2, 128, 128] [16, 2, 128, 128] [16, 2, 128, 128] [16, 2, 128, 128] [16, 2, 128, 128] ]
        #M_List V_List   M_list_gt V_List_gt

        # print(velocity.shape, len(SdefList), len(phiinvList), len(phiList), len(V_Z_List))
        # torch.Size([16, 2, 128, 128]) 7 7 7 7

        # 计算一些 visulaization 结果
        #all_videos_list, [16, 20, 7, 20, 20]  #7 frames
        ### Sampled results
        all_sample_vs_list = torch.stack(V_List, dim=2) #torch.Size([16, 2, 7, 128, 128])
        all_sample_vs_list = torch.cat((all_sample_vs_list, torch.zeros_like(all_sample_vs_list[:,0:1,:,:,:])), dim=1)
        all_sample_vs_list_list.append(all_sample_vs_list.detach().cpu().numpy())

        ### Numerical Solutions
        all_gtruth_vs_list = torch.stack(V_List_gt, dim=2) #torch.Size([16, 2, 7, 128, 128])
        all_gtruth_vs_list = torch.cat((all_gtruth_vs_list, torch.zeros_like(all_gtruth_vs_list[:,0:1,:,:,:])), dim=1)
        all_gtruth_vs_list_list.append(all_gtruth_vs_list.detach().cpu().numpy())
        
        ### EPDiff equation
        all_dec_vs_list = torch.stack(V_List_Dec, dim=2) #torch.Size([16, 2, 7, 128, 128])
        all_dec_vs_list = torch.cat((all_dec_vs_list, torch.zeros_like(all_dec_vs_list[:,0:1,:,:,:])), dim=1)
        all_dec_vs_list_list.append(all_dec_vs_list.detach().cpu().numpy())

        ### save phi and phiinv
        all_dec_phi_list = torch.stack(phiList, dim=2) #torch.Size([16, 2, 7, 128, 128])
        all_dec_phi_list_list.append(all_dec_phi_list.detach().cpu().numpy())
        
        all_dec_phiinv_list = torch.stack(phiinvList, dim=2) #torch.Size([16, 2, 7, 128, 128])
        all_dec_phiinv_list_list.append(all_dec_phiinv_list.detach().cpu().numpy())


        Channel = 1
        if "srcRGB" in sampled_batch:
            PlotRGB = True
        else:
            PlotRGB = False
        if PlotRGB == True:
            all_src = F.pad(srcRGB.permute(0,4,1,2,3), (2, 2, 2, 2))
            all_tar = F.pad(tarRGB.permute(0,4,1,2,3), (2, 2, 2, 2))
            all_src = rearrange(all_src, '(i j) c f h w -> c f (i h) (j w)', i = 4)
            all_tar = rearrange(all_tar, '(i j) c f h w -> c f (i h) (j w)', i = 4)
            src_path = f"{visjpg}/{i_batch_str}_GT_src.gif"
            tar_path = f"{visjpg}/{i_batch_str}_GT_tar.gif"

            ## sampled
            all_sample_dfm_gt_list = torch.cat(srcRGB_List_gt, dim=1) 
            all_sample_dfm_gt_list = all_sample_dfm_gt_list.permute(0,4,1,2,3)

            ## numerical solutions
            all_sample_dfm_list = torch.cat(srcRGB_List, dim=1) 
            all_sample_dfm_list = all_sample_dfm_list.permute(0,4,1,2,3)

            ## gdn
            all_sample_dfm_dec_list = torch.cat(srcRGB_List_Dec, dim=1) 
            all_sample_dfm_dec_list = all_sample_dfm_dec_list.permute(0,4,1,2,3)

            if SRGBdefList_TEST is not None:
                all_testdataset_dfm = torch.cat(SRGBdefList_TEST, dim=1) #torch.Size([16, 7, 128, 128, 3])
                all_testdataset_dfm = all_testdataset_dfm.permute(0,4,1,2,3)
                all_testdataset_dfm_list_list.append(all_testdataset_dfm.detach().cpu().numpy())


            all_sample_dfm_list_list.append(all_sample_dfm_list.detach().cpu().numpy())
            all_sample_dfm_gt_list_list.append(all_sample_dfm_gt_list.detach().cpu().numpy())
            all_sample_dfm_dec_list_list.append(all_sample_dfm_dec_list.detach().cpu().numpy())

             # all_testdataset_dfm_list = torch.cat(SdefList, dim=1) #torch.Size([16, 7, 128, 128])

            Channel = 3
        else:
            Channel = 1
            all_src = F.pad(src.unsqueeze(2), (2, 2, 2, 2))
            all_tar = F.pad(tar.unsqueeze(2), (2, 2, 2, 2))
            all_src = rearrange(all_src, '(i j) c f h w -> c f (i h) (j w)', i = 4)
            all_tar = rearrange(all_tar, '(i j) c f h w -> c f (i h) (j w)', i = 4)
            src_path = f"{visjpg}/{i_batch_str}_GT_src.gif"
            tar_path = f"{visjpg}/{i_batch_str}_GT_tar.gif"
        
            all_sample_vs_list = F.pad(all_sample_vs_list, (2, 2, 2, 2))
            all_gtruth_vs_list = F.pad(all_gtruth_vs_list, (2, 2, 2, 2))
            sampled_videos = F.pad(sampled_videos, (2, 2, 2, 2))
            # print(all_sample_vs_list.shape, all_gtruth_vs_list.shape, sampled_videos.shape) #torch.Size([16, 2, 9, 132, 132]) torch.Size([16, 2, 9, 132, 132]) torch.Size([16, 20, 9, 132, 132])
            
            # assert 3>444
            one_gif_sample = rearrange(all_sample_vs_list, '(i j) c f h w -> c f (i h) (j w)', i = 4)
            one_gif_gtruth = rearrange(all_gtruth_vs_list, '(i j) c f h w -> c f (i h) (j w)', i = 4)
            one_gif_sampled_videos = rearrange(sampled_videos, '(i j) c f h w -> c f (i h) (j w)', i = 4)
            # print(one_gif_sample.shape, one_gif_gtruth.shape,sampled_videos.shape) #orch.Size([3, 7, 512, 512]) torch.Size([3, 7, 512, 512])
        
            ### sampled
            all_sample_dfm_list = torch.stack(SdefList, dim=2) #torch.Size([16, 1, 7, 128, 128])
            ## numerical solution
            all_sample_dfm_gt_list = torch.stack(Sdef_List_gt, dim=2) #torch.Size([16, 1, 7, 128, 128])
            ## gdn
            all_sample_dfm_dec_list = torch.cat(Sdef_List_Dec, dim=1) 

            ''' all_sample_dfm_gt_list = torch.cat((src.unsqueeze(2), all_sample_dfm_gt_list), dim=2) #torch.Size([16, 1, 7, 128, 128])
            all_sample_dfm_list = torch.cat((src.unsqueeze(2), all_sample_dfm_list), dim=2) #torch.Size([16, 1, 7, 128, 128]) '''
            

            if SdefList_TEST is not None:
                all_testdataset_dfm_list = torch.stack(SdefList_TEST, dim=2)
                all_testdataset_dfm_list_list.append(all_testdataset_dfm_list.detach().cpu().numpy())

            all_sample_dfm_list_list.append(all_sample_dfm_list.detach().cpu().numpy())
            all_sample_dfm_gt_list_list.append(all_sample_dfm_gt_list.detach().cpu().numpy())
            all_sample_dfm_dec_list_list.append(all_sample_dfm_dec_list.detach().cpu().numpy())
            

            # print(all_sample_dfm_list.shape)

            # src                                               srcRGB tar tarRGB
            #([62, 1, 128, 128]) ([62, 1, 128, 128]) ([62, 1, 128, 128, 3]) ([62, 1, 128, 128, 3])
            # print(src.shape, tar.shape, srcRGB.shape, tarRGB.shape)




            


        #fixed_src = -1: regularity and v distribution
        # !=-1 confidence map: CDM
        """ if fixed_src != -1 and i_batch in [0,1]:
            pass
        elif fixed_src == -1 and i_batch in [2,3]:
            pass
        else:
            continue """

        # if fixed_src != -1 and i_batch >= 1:
        #     continue
        # elif fixed_src == -1 and i_batch > 4:
        #     continue
        # else:
        #     pass

        # if i_batch >= 3:
        #     continue
        
        

        all_sample_dfm_list = F.pad(all_sample_dfm_list, (2, 2, 2, 2))
        all_sample_dfm_gt_list = F.pad(all_sample_dfm_gt_list, (2, 2, 2, 2))

        one_gif_sampled_dfm = rearrange(all_sample_dfm_list, '(i j) c f h w -> c f (i h) (j w)', i = 4)
        one_gif_dfm_gt = rearrange(all_sample_dfm_gt_list, '(i j) c f h w -> c f (i h) (j w)', i = 4)
        # print(one_gif_sampled_dfm.shape, one_gif_dfm_gt.shape) #torch.Size([3, 7, 512, 512]) torch.Size([3, 7, 512, 512])
        # print(all_sample_dfm_list.shape, src.shape, tar.shape) #([16, 1, 8, 132, 132]) ([16, 1, 128, 128]) ([16, 1, 128, 128])
        # assert 3>444
        
        

        video_path_sample = f"{visjpg}/{i_batch_str}_velocity_sampled_GDN.gif"
        video_path_gtruth = f"{visjpg}/{i_batch_str}_velocity_EPDiff.gif"
        video_path_sampled_zv = f"{visjpg}/{i_batch_str}_velocity_latent_sampled.gif"
        video_path_dfm_gt = f"{visjpg}/{i_batch_str}_dfmSrc_EPDiff.gif"
        video_path_dfm = f"{visjpg}/{i_batch_str}_dfmSrc_sampled_GDN.gif"
        one_gif_sampled_phi_path = f"{visjpg}/{i_batch_str}_phi_sampled_GDN.gif"
        one_gif_sampled_phiinv_path = f"{visjpg}/{i_batch_str}_phiinv_sampled_GDN.gif"
        one_gif_sampled_phi_gt_path = f"{visjpg}/{i_batch_str}_phi_EPDiff.gif"
        one_gif_sampled_phiinv_gt_path = f"{visjpg}/{i_batch_str}_phiinv_EPDiff.gif"
        
        tar_sampled_path = f"{visjpg}/{i_batch_str}_tar_sampled.gif"

        # print(one_gif_sampled_dfm.shape) #[1, 8, 528, 528]
        # video_tensor_to_gif(one_gif_sampled_dfm[:,-1:], tar_sampled_path)  #[16, 1, 8, 132, 132]
        # assert 3>444



        # tensor_to_image(src[0,0,:,:].cpu().detach(), src_path)
        # tensor_to_image(tar[0,0,:,:].cpu().detach(), tar_path)
        # tensor_to_image(all_sample_dfm_list[0,0,-1,2:-2,2:-2].cpu().detach(), tar_sampled_path)


        #phiinvList  [16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128]
        #phiList     [16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128]

        # get_phi_img_arr
        ''' all_sample_phiinv_list = [get_phi_img_arr(item,scale=3,CNum=16,Config=Config) for item in phiinvList]
        all_sample_phiinv_list = torch.stack(all_sample_phiinv_list, dim=2)
        all_sample_phiinv_list = F.pad(all_sample_phiinv_list, (2, 2, 2, 2))
        one_gif_sampled_phiinv = rearrange(all_sample_phiinv_list, '(i j) c f h w -> c f (i h) (j w)', i = 4)
 '''

        ''' all_sample_phi_list = [get_phi_img_arr(item,scale=3,CNum=16,Config=Config) for item in phiList]
        # [16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128],[16, 2, 128, 128]
        all_sample_phi_list = torch.stack(all_sample_phi_list, dim=2) #torch.Size([16, 2, 7, 128, 128])
         #torch.Size([16, 2, 7, 128, 128])
        all_sample_phi_list = F.pad(all_sample_phi_list, (2, 2, 2, 2))
        one_gif_sampled_phi = rearrange(all_sample_phi_list, '(i j) c f h w -> c f (i h) (j w)', i = 4) '''
        
        # Phiinv_List_gt
        # Phi_List_gt
        ''' all_sample_phiinv_gt_list = [get_phi_img_arr(item,scale=3,CNum=16,Config=Config) for item in Phiinv_List_gt]
        all_sample_phi_gt_list = [get_phi_img_arr(item,scale=3,CNum=16,Config=Config) for item in Phi_List_gt]
        all_sample_phiinv_gt_list = torch.stack(all_sample_phiinv_gt_list, dim=2) #torch.Size([16, 2, 7, 128, 128])
        all_sample_phi_gt_list = torch.stack(all_sample_phi_gt_list, dim=2) #torch.Size([16, 2, 7, 128, 128])
        all_sample_phiinv_gt_list = F.pad(all_sample_phiinv_gt_list, (2, 2, 2, 2))
        all_sample_phi_gt_list = F.pad(all_sample_phi_gt_list, (2, 2, 2, 2))
        one_gif_sampled_phi_gt = rearrange(all_sample_phi_gt_list, '(i j) c f h w -> c f (i h) (j w)', i = 4)
        one_gif_sampled_phiinv_gt = rearrange(all_sample_phiinv_gt_list, '(i j) c f h w -> c f (i h) (j w)', i = 4) '''


        
        plot_result = True
        # if plot_result == True and i_batch in [0,1,2,3]:
        if plot_result == True:
            if fixed_src == -1:
                video_tensor_to_gif(all_src, src_path, c=Channel); video_tensor_to_gif(all_tar, tar_path, c=Channel)
            elif i_batch == 0:
                video_tensor_to_gif(all_src, src_path, c=Channel); video_tensor_to_gif(all_tar, tar_path, c=Channel)

            start_save_idx = -1
            video_tensor_to_gif(one_gif_sampled_dfm[:,start_save_idx:], video_path_dfm, c=Channel)
            ''' video_tensor_to_gif(one_gif_dfm_gt[:,start_save_idx:], video_path_dfm_gt, c=Channel) '''
            
            #velocity
            # video_tensor_to_gif(one_gif_sample[:,start_save_idx:], video_path_sample, c=1)
            # video_tensor_to_gif(one_gif_gtruth[:,start_save_idx:], video_path_gtruth, c=1)
            #latent velocity
            # video_tensor_to_gif(one_gif_sampled_videos[:,start_save_idx:], video_path_sampled_zv)
            
            # video_tensor_to_gif(one_gif_sampled_phi[:,start_save_idx:], one_gif_sampled_phi_path, c=3)
            ''' video_tensor_to_gif(one_gif_sampled_phiinv[:,start_save_idx:], one_gif_sampled_phiinv_path,c=3) '''
            # video_tensor_to_gif(one_gif_sampled_phi_gt[:,start_save_idx:], one_gif_sampled_phi_gt_path, c=3)
            ''' video_tensor_to_gif(one_gif_sampled_phiinv_gt[:,start_save_idx:], one_gif_sampled_phiinv_gt_path,c=3) '''



        # 7: print(len(SdefList), len(Sdef_List_gt), len(phiinvList), len(phiList), len(V_List), len(V_List_gt), len(Phiinv_List_gt), len(Dispinv_List))
        # print(SdefList[0].shape, Sdef_List_gt[0].shape, phiinvList[0].shape, phiList[0].shape, V_List[0].shape, V_List_gt[0].shape, Phiinv_List_gt[0].shape, Dispinv_List[0].shape)
        # [16, 1, 128, 128]  [16, 1, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]  [16, 2, 128, 128]

        # test_result['src'].append(src)
        # test_result['tar'].append(tar)
        # test_result['Sdef'].append(SdefList)
        # test_result['Sdef_gt'].append(Sdef_List_gt)
        # test_result['phiinv'].append(phiinvList)
        # test_result['phi'].append(phiList)
        # test_result['velocity'].append(V_List)
        # test_result['Phiinv_gt'].append(Phiinv_List_gt)
        # test_result['Dispinv'].append(Dispinv_List)
        # test_result['VList_gt'].append(V_List_gt)

        
    # wirht cond_text_list to textf_text_cond
    print(text_cond_list)
    for item in text_cond_list:
        textf_text_cond.write( item + '\n')
    textf_text_cond.close()


    sampled_videos_list_all = np.concatenate(sampled_videos_list, axis=0)   #[160, 20, 7, 16, 16]
    V_Z_List_NotUse_list_all = np.concatenate(V_Z_List_NotUse_list, axis=0) #[160, 20, 7, 16, 16]  
    src_cond_list_all = np.concatenate(src_cond_list, axis=0) #[160, 2, 20, 16, 16]
    all_sample_vs_list_list = np.concatenate(all_sample_vs_list_list, axis=0) #[160, 20, 7, 16, 16]
    all_gtruth_vs_list_list = np.concatenate(all_gtruth_vs_list_list, axis=0) #[160, 20, 7, 16, 16]
    all_dec_vs_list_list = np.concatenate(all_dec_vs_list_list, axis=0)

    all_dec_phi_list_list = np.concatenate(all_dec_phi_list_list, axis=0)
    all_dec_phiinv_list_list = np.concatenate(all_dec_phiinv_list_list, axis=0)


    for item in all_sample_dfm_list_list:
        print("all_sample_dfm_list_list:  ", item.shape)
    for item in all_sample_dfm_gt_list_list:
        print("all_sample_dfm_gt_list_list:  ",item.shape)
    for item in all_testdataset_dfm_list_list:
        print("all_testdataset_dfm_list_list:  ",item.shape)

    
    ## sampled
    all_sample_dfm_list_list = np.concatenate(all_sample_dfm_list_list, axis=0)   
    ## numerical solution
    all_sample_dfm_gt_list_list = np.concatenate(all_sample_dfm_gt_list_list, axis=0) 
    ## GDN
    all_sample_dfm_dec_list_list = np.concatenate(all_sample_dfm_dec_list_list, axis=0) 

    all_testdataset_dfm_list_list = np.concatenate(all_testdataset_dfm_list_list, axis=0) 
    print(all_sample_dfm_list_list.shape, all_sample_dfm_gt_list_list.shape, all_testdataset_dfm_list_list.shape)

    # KID_Value = [KID(realImg=srcRGB, sampledImg=srcRGB_gt) for (srcRGB, srcRGB_gt) in zip(all_sample_dfm_list_list, all_testdataset_dfm_list_list)]
    # FID_Value = [FID(realImg=srcRGB, sampledImg=srcRGB_gt) for (srcRGB, srcRGB_gt) in zip(all_sample_dfm_list_list, all_testdataset_dfm_list_list)]
    # print(f"KID_Value:\n {KID_Value}")
    # print(f"FID_Value:\n {FID_Value}")

    # f"{visjpg}/{TestDateStr}_velocity_sampled_GDN.gif"
    totalnum = sampled_videos_list_all.shape[0]
    #detect folder exist or not
    if not os.path.exists(f'{visjpg}/npy'):
        os.makedirs(f'{visjpg}/npy')
    
    ## sampled
    #For confidence map
    np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_dfm_sampled.npy', all_sample_dfm_list_list)
    #For Geodesice Path
    np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_v_s_sampled.npy', all_sample_vs_list_list)

    np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_phi_dec.npy', all_dec_phi_list_list)
    np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_phiinv_dec.npy', all_dec_phiinv_list_list)


    #For KID FID IS FVD and Regularity MSE error
    if fixed_src == -1:
        # np.save(f'{visjpg}/npy/{TestDateStr}_{totalnum}_v_z_sampled.npy', sampled_videos_list_all)
        # np.save(f'{visjpg}/npy/{TestDateStr}_{totalnum}_v_z_encoder.npy', V_Z_List_NotUse_list_all)
        # np.save(f'{visjpg}/npy/{TestDateStr}_{totalnum}_src_cond.npy', src_cond_list_all)

        ## numerical solution
        np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_dfm_gt_epdiff.npy', all_sample_dfm_gt_list_list)
        np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_v_s_gt_epdiff.npy', all_gtruth_vs_list_list)

        np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_dfm_dec_epdiff.npy', all_sample_dfm_dec_list_list)
        np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_v_s_dec_epdiff.npy', all_dec_vs_list_list)

        np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_phi_dec.npy', all_dec_phi_list_list)
        np.save(f'{visjpg}/npy/{TestDateStr}_{fixed_src}_{totalnum}_phiinv_dec.npy', all_dec_phiinv_list_list)

        np.save(f'{visjpg}/npy/{TestDateStr}_{totalnum}_dfm_testdataset.npy', all_testdataset_dfm_list_list)
    
    assert 3>333
    print(sampled_videos_list_all.shape, V_Z_List_NotUse_list_all.shape, src_cond_list_all.shape, all_sample_vs_list_list.shape, all_gtruth_vs_list_list.shape)

    
    textf_GDN_DiFuS.close()
    textf_EPDiff.close()
