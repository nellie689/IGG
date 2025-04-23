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
from Util.EpdiffLib import Grad
import torch.nn.functional as F
from timeit import default_timer
from Util.utils import LpLoss, get_source_gradient
from tensorboardX import SummaryWriter
from Util.utils import MgridVelocity, Mgridplot, get_text_list, get_text_list_from_time
import matplotlib.pyplot as plt


criterion = nn.MSELoss()
myloss = LpLoss(size_average=False)
regloss = Grad('l2').loss

def trainer_GDN(Config, model):
    random.seed(Config["general"]["seed"])
    np.random.seed(Config["general"]["seed"])
    torch.manual_seed(Config["general"]["seed"])
    torch.cuda.manual_seed(Config["general"]["seed"])

    # from cosh.get_coshrem_args import get_xform
    # xforms = get_xform()

    snapshot_path = Config["general"]["snapshot_path"]
    # TSteps = Config["general"]["num_steps"]
    # WSimi = Config["LossWeightSvf"]["WSimi"]
    # WReg = Config["LossWeightSvf"]["WReg"]
    LR = Config["LStrategy"]["Registration"]["lr"]
    weight_decay = Config["general"]["WD"]

    #loss weights
    WSimi = Config["LossWeightLddmm"]["WSimi"]
    WReg = Config["LossWeightLddmm"]["WReg"]
    sigma = Config["LossWeightLddmm"]["sigma"]
    WEPDiff_Mse = Config["LossWeightLddmm"]["WEPDiff_Mse"]
    WEPDiff_Relative = Config["LossWeightLddmm"]["WEPDiff_Relative"]
    # print("WSimi", WSimi, "WReg", WReg, "sigma", sigma, "WEPDiff_Mse", WEPDiff_Mse, "WEPDiff_Relative", WEPDiff_Relative)
    # WSimi 0.5 WReg 10.0 sigma 0.01 WEPDiff_Mse 100.0 WEPDiff_Relative 100.0
    TSteps = Config['general']['num_steps']
   
    

    writerdir = snapshot_path.split('/'); writerdir.insert(-1,'log'); writerdir='/'.join(writerdir)
    writer = SummaryWriter(writerdir)

    # print(model)
    # assert 3>333
    
    optimizer_regis = torch.optim.Adam([ {'params': model.parameters(), 'lr': LR}], weight_decay=weight_decay)
    # 优化 encoder + SelfAttention2
    if hasattr(model, 'SelfAttention'):
        optimizer_encoder_attention = torch.optim.Adam(list(model.encoder.parameters())+list(model.SelfAttention.parameters())+list(model.remain.parameters()), lr=LR, weight_decay=weight_decay)
    else:
        optimizer_encoder_attention = torch.optim.Adam(list(model.encoder.parameters())+list(model.remain.parameters()), lr=LR, weight_decay=weight_decay)

    if Config["general"]["SepFNODeCoder"] == "Yes":
        # 优化 model_v
        optimizer_model_v = torch.optim.Adam(
            list(model.model_v.parameters()) + list(model.remain_model_v.parameters()), 
            lr=LR, 
            weight_decay=weight_decay
        )
        if hasattr(model, 'SelfAttention'):
            optimizer_joint = torch.optim.Adam(
                list(model.model_v.parameters()) + list(model.remain_model_v.parameters()) + 
                list(model.encoder.parameters()) + list(model.SelfAttention.parameters()) +list(model.remain.parameters()), 
                lr=LR, 
                weight_decay=weight_decay
            )
        else:
            optimizer_joint = torch.optim.Adam(
                list(model.model_v.parameters()) + list(model.remain_model_v.parameters()) + 
                list(model.encoder.parameters()) + list(model.remain.parameters()), 
                lr=LR, 
                weight_decay=weight_decay
            )
    else:
        optimizer_model_v = torch.optim.Adam(
            list(model.model_v.parameters()), 
            lr=LR, 
            weight_decay=weight_decay
        )
        if hasattr(model, 'SelfAttention'):
            optimizer_joint = torch.optim.Adam(
                list(model.model_v.parameters()) + list(model.encoder.parameters()) + 
                list(model.SelfAttention.parameters()) + list(model.remain.parameters()), 
                lr=LR, 
                weight_decay=weight_decay
            )
        else:
            optimizer_joint = torch.optim.Adam(
                list(model.model_v.parameters()) + list(model.encoder.parameters()) + list(model.remain.parameters()), 
                lr=LR, 
                weight_decay=weight_decay
            )

        

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(Config))
    
    if Config["DataSet"]["dataName"] in ["OASIS32D"] or "Plant" in Config["DataSet"]["dataName"]:
        trainloader, text_data = loadTrain(Config); testloader, _ = loadTest(Config)
    else:
        trainloader = loadTrain(Config); testloader = loadTest(Config)
    
    max_epoch = Config["LStrategy"]["max_epochs"]
    max_iterations = max_epoch * len(trainloader)  # 8000=500*16     max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    iter_num = 0
    loss1_min = 999999
    loss_min = 999999
    

    altarFlag = "Register" #"Joint" "Fno"
    altarFlag = "Fno"
    if altarFlag == "Register":
        model.freeze_model_v()
        model.unfreeze_encoder_and_attention()
    if altarFlag == "Fno": #load pre-trained model of Register
        path_model_Register = snapshot_path+ "/Register_{}.pth".format(Config["LStrategy"]["max_epochs"] - 1)
        path_model_ATT = snapshot_path+ "/ATT_{}.pth".format(Config["LStrategy"]["max_epochs"] - 1)
        #load the model
        checkpoint = torch.load(path_model_Register)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.remain.load_state_dict(checkpoint['remain'])
        # model.SelfAttention.load_state_dict(checkpoint['SelfAttention'])
        if hasattr(model, 'SelfAttention'):
            checkpoint = torch.load(path_model_ATT)
            model.SelfAttention.load_state_dict(checkpoint['SelfAttention'])
        model.unfreeze_model_v()
        model.freeze_encoder_and_attention()


    for epoch_num in iterator:
        #0.0005->0.00025  ->0.000125 ->0.0000625 ->0.00003125
        #0-3000->3000-6000->6000-9000->9000-12000->12000-20000
        # if (epoch_num)%3000 == 0:
        #     if optimizer_regis.param_groups[0]['lr'] > 0.00005:
        #         optimizer_regis.param_groups[0]['lr'] = optimizer_regis.param_groups[0]['lr'] * 0.5
        if (epoch_num)%6000 == 0:
            # if optimizer_regis.param_groups[0]['lr'] > 0.0001:
            #     optimizer_regis.param_groups[0]['lr'] = optimizer_regis.param_groups[0]['lr'] * 0.5
            #optimizer_encoder_attention optimizer_model_v optimzer_joint
            if optimizer_encoder_attention.param_groups[0]['lr'] > 0.0001:
                optimizer_encoder_attention.param_groups[0]['lr'] = optimizer_encoder_attention.param_groups[0]['lr'] * 0.5
            # if optimizer_model_v.param_groups[0]['lr'] > 0.0001:
            #     optimizer_model_v.param_groups[0]['lr'] = optimizer_model_v.param_groups[0]['lr'] * 0.5
            # if optimzer_joint.param_groups[0]['lr'] > 0.0001:
            #     optimzer_joint.param_groups[0]['lr'] = optimzer_joint.param_groups[0]['lr'] * 0.5


                
        # if (epoch_num+1) == 902:
        #     assert 1>999
        if epoch_num %100 == 0:
            model.graph_flag = True; model.gradlist = []
        start_time = default_timer()
        model.train()
        train_loss = {"total_loss":[], "loss_regis":[], "similarity":[], "regularity":[], "Fno": [], "Fno_mse": [], "dice": []}
        test_loss = {"total_loss":[], "loss_regis":[], "similarity":[], "regularity":[], "Fno": [], "Fno_mse": [], "dice": [], "v": [], "v_mse": [], "joint": []}
        

        for i_batch, sampled_batch in enumerate(trainloader):
            model.graph_flag2 = True
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
                
            else:  #["PlantGray", "PlantGray2"]
                src, tar = sampled_batch['src'], sampled_batch['tar']
                b,b,w,h = src.shape
                src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]


            
            B,C,W,H = src.shape
            

            
            iter_num = iter_num + 1
            if altarFlag == "Joint":
                Sdef, V_List, V_List_gt, M_List = model(torch.cat((src,tar),dim=1),altarFlag=altarFlag)
                loss_v = 0
                loss_v_mse = 0
                for t in range(1, TSteps):
                    VT = V_List[t]; VT_gt = V_List_gt[t]
                    loss_v += myloss(VT, VT_gt)
                    loss_v_mse += criterion(VT,VT_gt)
                loss_v /= B
                loss_v /= (TSteps-1)
                loss_v_mse /= (TSteps-1)
                
                V0 = V_List[0]; M0 = M_List[0]
                loss1 = criterion(tar,Sdef)
                loss2 =(V0*M0).sum() / (src.numel())
                loss_regis = WSimi * loss1/(sigma*sigma) + WReg * loss2
                # loss1 = criterion(tar,Sdef)
                # loss2 = regloss(None, velocity)
                # loss = loss1*WSimi + WReg*loss2

                loss_joint = loss_regis + WEPDiff_Mse*loss_v_mse + WEPDiff_Relative*loss_v

                optimizer_joint.zero_grad()
                loss_joint.backward()
                optimizer_joint.step()
                
                writer.add_scalar('info/train_totalloss', loss_regis.item(), iter_num)
                writer.add_scalar('info/train_similarity', loss1.item(), iter_num)
                writer.add_scalar('info/train_regularity', loss2.item(), iter_num)
                writer.add_scalar('info/train_v', loss_v.item(), iter_num)
                writer.add_scalar('info/train_v_mse', loss_v_mse.item(), iter_num)
                writer.add_scalar('info/train_joint', loss_joint.item(), iter_num)
                logging.info(f'JoingTrain~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} src {sampled_batch["src"].shape} totalloss {loss_regis.item()} similarity {loss1.item()} regularity {loss2.item()} \
                            \n v {loss_v.item()} v_mse {loss_v_mse.item()} joint {loss_joint.item()}')
            elif altarFlag == "Register":
                Sdef, V0, M0 = model(torch.cat((src,tar),dim=1),altarFlag=altarFlag)
                loss1 = criterion(tar,Sdef)
                loss2 =(V0*M0).sum() / (src.numel())
                loss_regis = WSimi * loss1/(sigma*sigma) + WReg * loss2

                optimizer_encoder_attention.zero_grad()
                loss_regis.backward()
                optimizer_encoder_attention.step()

                writer.add_scalar('info/train_totalloss', loss_regis.item(), iter_num)
                writer.add_scalar('info/train_similarity', loss1.item(), iter_num)
                writer.add_scalar('info/train_regularity', loss2.item(), iter_num)
                logging.info(f'RegisterTrain~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} src {sampled_batch["src"].shape} totalloss {loss_regis.item()} similarity {loss1.item()} regularity {loss2.item()}')
            elif altarFlag == "Fno":
                V_List, V_List_gt = model(torch.cat((src,tar),dim=1),altarFlag=altarFlag)
                loss_v = 0
                loss_v_mse = 0
                for t in range(1, TSteps):
                    VT = V_List[t]; VT_gt = V_List_gt[t]
                    loss_v += myloss(VT, VT_gt)
                    loss_v_mse += criterion(VT,VT_gt)
                loss_v /= B
                loss_v /= (TSteps-1)
                loss_v_mse /= (TSteps-1)
                FNO_loss = 10.0*loss_v_mse + 10.0*loss_v

                optimizer_model_v.zero_grad()
                FNO_loss.backward()
                # for param in model.remain.parameters():
                #     print(param.grad)
                # for param in model.remain_model_v.parameters():
                #     print(param.grad, param.requires_grad)
                # for param in model.model_v.parameters():
                #     print(torch.mean(param.grad))
                # assert 4>444
                optimizer_model_v.step()

                writer.add_scalar('info/train_v', loss_v.item(), iter_num)
                writer.add_scalar('info/train_v_mse', loss_v_mse.item(), iter_num)
                writer.add_scalar('info/train_FNO', FNO_loss.item(), iter_num)
                logging.info(f'FnoTrain~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} src {sampled_batch["src"].shape} v {loss_v.item()} v_mse {loss_v_mse.item()} FNO {FNO_loss.item()}')


        with torch.no_grad():
            model.graph_flag2 = False
            model.graph_flag = False
            model.eval()
            for i_batch, sampled_batch in enumerate(testloader): 
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
                else:
                    src, tar = sampled_batch['src'], sampled_batch['tar']
                    b,b,w,h = src.shape
                    src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]

                
            
                # v_seq_full_dim, v_seq_gt_epdiff, Sdef, Sdfm_binary, m_seq_full_dim, phiinv_pred = model(torch.cat((src,tar),dim=1))
                ''' Sdef, velocity, phiinv = model(torch.cat((src,tar),dim=1))
                loss1 = criterion(tar,Sdef)
                loss2 = regloss(None, velocity)
                loss = loss1*WSimi + WReg*loss2 '''

                
                B,C,W,H = src.shape
                if altarFlag == "Joint":
                    Sdef, V_List, V_List_gt, M_List = model(torch.cat((src,tar),dim=1))
                    loss_v = 0
                    loss_v_mse = 0
                    for t in range(1, TSteps):
                        VT = V_List[t]; VT_gt = V_List_gt[t]
                        loss_v += myloss(VT, VT_gt)
                        loss_v_mse += criterion(VT,VT_gt)
                    loss_v /= B
                    loss_v /= (TSteps-1)
                    loss_v_mse /= (TSteps-1)
                    
                    V0 = V_List[0]; M0 = M_List[0]
                    loss1 = criterion(tar,Sdef)
                    loss2 =(V0*M0).sum() / (src.numel())
                    loss_regis = WSimi * loss1/(sigma*sigma) + WReg * loss2
                    # loss1 = criterion(tar,Sdef)
                    # loss2 = regloss(None, velocity)
                    # loss = loss1*WSimi + WReg*loss2
                    loss_joint = loss_regis + WEPDiff_Mse*loss_v_mse + WEPDiff_Relative*loss_v

                    test_loss['total_loss'].append(loss_regis.item())
                    test_loss['similarity'].append(loss1.item())
                    test_loss['regularity'].append(loss2.item())
                    test_loss['v'].append(loss_v.item())
                    test_loss['v_mse'].append(loss_v_mse.item())
                    test_loss['joint'].append(loss_joint.item())
                
                elif altarFlag == "Register":
                    Sdef, V0, M0 = model(torch.cat((src,tar),dim=1),altarFlag=altarFlag)
                    loss1 = criterion(tar,Sdef)
                    loss2 =(V0*M0).sum() / (src.numel())
                    loss_regis = WSimi * loss1/(sigma*sigma) + WReg * loss2
                    test_loss['total_loss'].append(loss_regis.item())
                    test_loss['similarity'].append(loss1.item())
                    test_loss['regularity'].append(loss2.item())
                elif altarFlag == "Fno":
                    V_List, V_List_gt = model(torch.cat((src,tar),dim=1),altarFlag=altarFlag)
                    loss_v = 0
                    loss_v_mse = 0
                    for t in range(1, TSteps):
                        VT = V_List[t]; VT_gt = V_List_gt[t]
                        loss_v += myloss(VT, VT_gt)
                        loss_v_mse += criterion(VT,VT_gt)
                    loss_v /= B
                    loss_v /= (TSteps-1)
                    loss_v_mse /= (TSteps-1)
                    FNO_loss = 10.0*loss_v_mse + 10.0*loss_v
                    test_loss['v'].append(loss_v.item())
                    test_loss['v_mse'].append(loss_v_mse.item())
                    test_loss['Fno'].append(FNO_loss.item())

                
            if altarFlag == "Joint":
                writer.add_scalar('info/test_totalloss', np.mean(test_loss['total_loss']), iter_num)
                writer.add_scalar('info/test_similarity', np.mean(test_loss['similarity']), iter_num)
                writer.add_scalar('info/test_regularity', np.mean(test_loss['regularity']), iter_num)
                writer.add_scalar('info/test_v', np.mean(test_loss['v']), iter_num)
                writer.add_scalar('info/test_v_mse', np.mean(test_loss['v_mse']), iter_num)
                writer.add_scalar('info/test_joint', np.mean(test_loss['joint']), iter_num)
                logging.info(f"JointTest~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} totalloss {np.mean(test_loss['total_loss'])} \
                                similarity {np.mean(test_loss['similarity'])} regularity {np.mean(test_loss['regularity'])} \
                                \n v {np.mean(test_loss['v'])} v_mse {np.mean(test_loss['v_mse'])} joint {np.mean(test_loss['joint'])}")
            elif altarFlag == "Register":
                writer.add_scalar('info/test_totalloss', np.mean(test_loss['total_loss']), iter_num)
                writer.add_scalar('info/test_similarity', np.mean(test_loss['similarity']), iter_num)
                writer.add_scalar('info/test_regularity', np.mean(test_loss['regularity']), iter_num)
                logging.info(f"VoxelmorphTest~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} totalloss {np.mean(test_loss['total_loss'])} \
                                similarity {np.mean(test_loss['similarity'])} regularity {np.mean(test_loss['regularity'])}")    
            elif altarFlag == "Fno":
                writer.add_scalar('info/test_v', np.mean(test_loss['v']), iter_num)
                writer.add_scalar('info/test_v_mse', np.mean(test_loss['v_mse']), iter_num)
                logging.info(f"FnoTest~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} v {np.mean(test_loss['v'])} v_mse {np.mean(test_loss['v_mse'])} FNO {np.mean(test_loss['Fno'])}")

        if altarFlag == "Joint":
            if((epoch_num+1)%200)==0:
                try:
                    os.remove(path_model_joint)
                except:
                    pass
                path_model_joint = snapshot_path+ f'/registration_SepFNODeCoder{Config["general"]["SepFNODeCoder"]}_{epoch_num}.pth'
                if Config["general"]["SepFNODeCoder"] == "Yes":
                    torch.save({
                        'encoder': model.encoder.state_dict(),
                        'model_v': model.model_v.state_dict(),
                        'remain_model_v': model.remain_model_v.state_dict(),
                        'remain': model.remain.state_dict(),
                    }, path_model_joint)
                elif Config["general"]["SepFNODeCoder"] == "No":
                    torch.save({
                        'encoder': model.encoder.state_dict(),
                        
                        'model': model.model.state_dict(),
                        'remain': model.remain.state_dict(),
                    }, path_model_joint)
                # 'SelfAttention': model.SelfAttention.state_dict(),
                if hasattr(model, 'SelfAttention'):
                    try :
                        os.remove(path_model_ATT)
                    except:
                        pass
                    path_model_ATT = snapshot_path+ "/ATT_{}.pth".format(epoch_num)
                    torch.save({
                        'SelfAttention': model.SelfAttention.state_dict()
                    }, path_model_ATT)
        elif altarFlag == "Register":
            if((epoch_num+1)%200)==0:
                try:
                    os.remove(path_model_Register)
                except:
                    pass
                path_model_Register = snapshot_path+ "/Register_{}.pth".format(epoch_num)
                torch.save({
                    'encoder': model.encoder.state_dict(),
                    'remain': model.remain.state_dict(),
                }, path_model_Register)

                # 'SelfAttention': model.SelfAttention.state_dict(),
                if hasattr(model, 'SelfAttention'):
                    try :
                        os.remove(path_model_ATT)
                    except:
                        pass
                    path_model_ATT = snapshot_path+ "/ATT_{}.pth".format(epoch_num)
                    torch.save({
                        'SelfAttention': model.SelfAttention.state_dict()
                    }, path_model_ATT)

        elif altarFlag == "Fno":
            ModesFno = Config["general"]["ModesFno"]
            WidthFno = Config["general"]["WidthFno"]

            if((epoch_num+1)%200)==0:
                try:
                    os.remove(path_model_Fno)
                except:
                    pass
                path_model_Fno = snapshot_path+ f'/Fno_{ModesFno}_{WidthFno}_SepFNODeCoder{Config["general"]["SepFNODeCoder"]}_{epoch_num}.pth'
                
                if Config["general"]["SepFNODeCoder"] == "Yes":
                    torch.save({
                        'model_v': model.model_v.state_dict(),
                        'remain_model_v': model.remain_model_v.state_dict(),
                    }, path_model_Fno)
                elif Config["general"]["SepFNODeCoder"] == "No":
                    torch.save({
                        'model_v': model.model_v.state_dict(),
                    }, path_model_Fno)
            

        end_time = default_timer()
        logging.info(f"Epoch {epoch_num} finished. Time: {end_time-start_time} seconds\n")


def trainer_GDN_DiFuS(Config, model):
    # frozen model's parameters
    model.freeze_model(model)
    random.seed(Config["general"]["seed"])
    np.random.seed(Config["general"]["seed"])
    torch.manual_seed(Config["general"]["seed"])
    torch.cuda.manual_seed(Config["general"]["seed"])

    # from cosh.get_coshrem_args import get_xform
    # xforms = get_xform()

    snapshot_path = Config["general"]["snapshot_path"]
    # TSteps = Config["general"]["num_steps"]
    # WSimi = Config["LossWeightSvf"]["WSimi"]
    # WReg = Config["LossWeightSvf"]["WReg"]
    LR = Config["LStrategy"]["Registration"]["lr"]
    weight_decay = Config["general"]["WD"]
    null_src_cond_prob = Config["general"]["null_src_cond_prob"]
    null_cond_prob = Config["general"]["null_cond_prob"]
    


    #loss weights
    WSimi = Config["LossWeightLddmm"]["WSimi"]
    WReg = Config["LossWeightLddmm"]["WReg"]
    sigma = Config["LossWeightLddmm"]["sigma"]
    WEPDiff_Mse = Config["LossWeightLddmm"]["WEPDiff_Mse"]
    WEPDiff_Relative = Config["LossWeightLddmm"]["WEPDiff_Relative"]
    # print("WSimi", WSimi, "WReg", WReg, "sigma", sigma, "WEPDiff_Mse", WEPDiff_Mse, "WEPDiff_Relative", WEPDiff_Relative)
    # WSimi 0.5 WReg 10.0 sigma 0.01 WEPDiff_Mse 100.0 WEPDiff_Relative 100.0
    TSteps = Config['general']['num_steps']
   

    writerdir = snapshot_path.split('/'); writerdir.insert(-1,'log'); writerdir='/'.join(writerdir)
    writer = SummaryWriter(writerdir)

    # print(model)
    # assert 3>333
    
    optimizer_regis = torch.optim.Adam([ {'params': model.parameters(), 'lr': LR}], weight_decay=weight_decay)
    

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(Config))
    
    # assert 1>123
    if Config["DataSet"]["dataName"] in ["OASIS32D"] or "Plant" in Config["DataSet"]["dataName"]:
        trainloader, text_data = loadTrain(Config); testloader, _ = loadTest(Config)
        use_bert_text_cond = True
    else:
        trainloader = loadTrain(Config); testloader = loadTest(Config)
        use_bert_text_cond = False

    # use_bert_text_cond = False  #Nellie
    max_epoch = Config["LStrategy"]["max_epochs"]
    max_iterations = max_epoch * len(trainloader)  # 8000=500*16     max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    iter_num = 0
    loss1_min = 999999
    loss_min = 999999

    
    
    from networks.video_diffusion_pytorch import Trainer_with_GDN, Unet3D, GaussianDiffusion, Trainer

    diffusion = GaussianDiffusion(
        Unet3D(
            dim = 32,
            dim_mults = (1, 2, 4),
            channels = 20+Config["general"]["ConditionCH"],
            out_dim = 20,
            use_bert_text_cond = use_bert_text_cond,  # this must be set to True to auto-use the bert model dimensions
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
        train_lr = LR,
        save_and_sample_every = 1000,     #save every 5000 epoches
        # save_and_sample_every = 1,
        # train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                        # turn on mixed precision
        results_folder = snapshot_path,
    )


    #load pre-trained model of Register
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
    # print(model)
    # assert 3>111

    

    begin_time = default_timer()
    for epoch_num in iterator:
        save_model_flag = True
        start_time = default_timer()
        test_loss = {"loss":[]}
        
        for i_batch, sampled_batch in enumerate(trainloader):
            text_cond = None
            src, tar = sampled_batch['src'], sampled_batch['tar']
            b,b,w,h = src.shape
            src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]

            if 'textidx' in sampled_batch:
                textidx = sampled_batch['textidx']
                textidx = textidx.reshape(-1)
                text_cond = get_text_list(textidx, text_data)
                print(1111111111111111111111)
            
            if "src_time" in sampled_batch:
                src_time = sampled_batch['src_time']; src_time = src_time.reshape(-1)
                tar_time = sampled_batch['tar_time']; tar_time = tar_time.reshape(-1)
                text_cond = get_text_list_from_time(src_time, tar_time)
                print(2222222222222222222222)
            

            V_Z_List = model(torch.cat((src,tar),dim=1))  #[100, 20, 7, 16, 16]


            if "Plant" in Config["DataSet"]["dataName"]:
                src_cond = get_source_gradient(src) * 5
            elif Config["DataSet"]["dataName"] in ["Shape"]:
                src_cond = get_source_gradient(src) * 2
            else:
                src_cond = get_source_gradient(src) * 1
            # print(src_cond.shape)   #[48, 2, 128, 128]

            src_cond = src_cond.unsqueeze(2)[:,:,:,::8,::8]
            src_cond = src_cond.repeat(1,1,7,1,1)
            # print(src_cond.shape)   #[48, 2, 7, 16, 16]
            # assert 2>111
            null_cond_prob_ = null_cond_prob
            null_src_cond_prob_ = null_src_cond_prob


            loss = diffusion_trainer.train(V_Z_List, src_cond = src_cond, Config=Config, epoch = epoch_num, save_model_flag=save_model_flag, text_cond = text_cond, null_src_cond_prob=null_src_cond_prob_, null_cond_prob=null_cond_prob_)
            iter_num = iter_num + 1


            


            if save_model_flag:
                writer.add_scalar('info/train_loss', loss.item(), epoch_num)
                logging.info(f'Train_GDN_DiFuS~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} src {sampled_batch["src"].shape} loss {loss.item()}')
                save_model_flag = False



        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(testloader): 
                text_cond = None
                src, tar = sampled_batch['src'], sampled_batch['tar']
                b,b,w,h = src.shape
                src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]

                if 'textidx' in sampled_batch:
                    textidx = sampled_batch['textidx']
                    textidx = textidx.reshape(-1)
                    text_cond = get_text_list(textidx, text_data)

                if "src_time" in sampled_batch:
                    # print(sampled_batch['src_time'].shape, sampled_batch['tar_time'].shape)
                    src_time = sampled_batch['src_time']; src_time = src_time.reshape(-1)
                    tar_time = sampled_batch['tar_time']; tar_time = tar_time.reshape(-1)
                    # print(src_time.shape, tar_time.shape)
                    text_cond = get_text_list_from_time(src_time, tar_time)
                    # print(text_cond)
                
                V_Z_List = model(torch.cat((src,tar),dim=1))
                if Config["DataSet"]["dataName"] in ["Mnist", "BullEye"] or "Plant" in Config["DataSet"]["dataName"]:
                    src_cond = get_source_gradient(src) * 5
                elif Config["DataSet"]["dataName"] in ["Shape"]:
                    src_cond = get_source_gradient(src) * 2
                else:
                    src_cond = get_source_gradient(src) * 1
                src_cond = src_cond.unsqueeze(2)[:,:,:,::8,::8]
                src_cond = src_cond.repeat(1,1,7,1,1)

                loss = diffusion_trainer.test(V_Z_List, src_cond = src_cond, text_cond = text_cond)
                test_loss['loss'].append(loss.item())

            writer.add_scalar('info/test_loss', np.mean(test_loss['loss']), epoch_num)

            end_time = default_timer()
            logging.info(f"Test_GDN_DiFuS~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} totalloss {np.mean(test_loss['loss'])}   time: {end_time-start_time} seconds  total-time: {(end_time-begin_time)/3600} hours\n")

