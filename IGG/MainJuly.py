import os
# from utilities3 import *
# from uEpdiff_ipmi import *
import argparse
import json
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from trainer import trainer_GDN, trainer_GDN_DiFuS
from tester import tester_GDN, tester_GDN_DiFuS
from networks.simli_vm_July import NetJuly
from Util.base import load_config


from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--server', type=str, default="My", help='My')
parser.add_argument('--mode', type=str, default="test", help='My')
parser.add_argument('--max_epochs', type=int, default=100, help='My')
parser.add_argument('--crop_size', type=int, default=16, help='My')
parser.add_argument('--module_name', type=str, default="SYMNet16", help='module_name')
parser.add_argument('--WReg', type=float, default=0.03, help='WReg')
parser.add_argument('--WSimi', type=float, default=0.5, help='WSimi')
parser.add_argument('--shootingType', type=str, default="svf", help='lddmm')
parser.add_argument('--activation', type=str, default="leakyrelu", help='zreLU')
#leakyrelu
parser.add_argument('--leakyrelu', type=float, default=0.02, help='leakyrelu')

parser.add_argument('--sub_version', type=str, default="000", help='sub_version')


# alpha=1.0 gamma=0.5 power
parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--power', type=float, default=2.0, help='power')
#num_steps
parser.add_argument('--num_steps', type=int, default=7, help='num_steps')
#train_batch_size
parser.add_argument('--train_batch_size', type=int, default=32, help='train_batch_size')

# sigma=0.01
# WReg=10
# WEPDiff_Mse=100
# WEPDiff_Relative=100
parser.add_argument('--sigma', type=float, default=0.01, help='sigma')
parser.add_argument('--WEPDiff_Mse', type=float, default=100, help='WEPDiff_Mse')
parser.add_argument('--WEPDiff_Relative', type=float, default=100, help='WEPDiff_Relative')


parser.add_argument('--blocks', type=int, default=2, help='blocks of attentions')
parser.add_argument('--WD', type=float, default=0, help='Weight decay')

parser.add_argument('--SYMCH', type=float, default=8, help='SYMCH')

#LR
parser.add_argument('--LR', type=float, default=0.001, help='lr')

#DifuS_version
parser.add_argument('--DifuS_version', type=str, default="way1", help='way1')
#Condition
parser.add_argument('--Condition', type=str, default="01", help='only on the input')
#fixed_src
parser.add_argument('--fixed_src', type=int, default=-1, help='fixed_src')
#-dataName $dataName --data_base_size $data_base_size --train_type $train_type --test_type $test_type
parser.add_argument('--dataName', type=str, default="BullEye", help='BullEye')
parser.add_argument('--data_base_size', type=int, default=128, help='128')
parser.add_argument('--train_type', type=str, default="BullEye", help='BullEye')
parser.add_argument('--test_type', type=str, default="BullEye", help='BullEye')
#Mnist_Tag Shape_Tag
parser.add_argument('--Mnist_Tag', type=int, default=4, help='4')
parser.add_argument('--Shape_Tag', type=str, default="heart", help='heart')
#level
parser.add_argument('--level', type=int, default=0, help='0')
#LSize
parser.add_argument('--LSize', type=int, default=16, help='0')
#SepFNODeCoder
parser.add_argument('--SepFNODeCoder', type=str, default=0, help='Yes')
#loadModel
parser.add_argument('--loadModelAtBegin', type=str, default="True", help='Yes')
#ModesFno
parser.add_argument('--ModesFno', type=int, default=4, help='1')
#WidthFno
parser.add_argument('--WidthFno', type=int, default=20, help='64')
#PlantType
parser.add_argument('--PlantType', type=str, default="Normal", help='Normal')
#USE_AUG
parser.add_argument('--USE_AUG', type=str, default="None", help='True')
#null_src_cond_prob null_cond_prob
parser.add_argument('--null_src_cond_prob', type=float, default=0.0, help='0.0')
parser.add_argument('--null_cond_prob', type=float, default=0.0, help='0.0')
## cond_scale # src_cond_scale=2.0
parser.add_argument('--cond_scale', type=float, default=1.5, help='1.0')
parser.add_argument('--src_cond_scale', type=float, default=2.0, help='1.0')

# args = get_all_mixture_args(parser)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if args.server == "CS":
        HOME = "/p/mmcardiac/nellie"
    elif args.server == "Ri":
        HOME = "/scratch/bsw3ac/nellie"
    else:
        HOME = "/home/nellie"
    # os.path.abspath(HOME+f'/code/cvpr/ComplexNet/My2D')
        
    # 加载默认配置
    BaseConfig = load_config(f'{HOME}/code/cvpr/ComplexNet/My2D/Config/BaseConfig.json')
    #add home to BaseConfig["general"]

    Config = BaseConfig
    #server
    Config["general"]["server"] = args.server
    Config["general"]["HOME"] = HOME
    Config["general"]["mode"] = args.mode
    Config["LStrategy"]["max_epochs"] = args.max_epochs
    Config["DataSet"]["crop_size"] = args.crop_size
    Config["general"]["module_name"] = args.module_name
    Config["general"]["shootingType"] = args.shootingType
    Config["general"]["activation"] = args.activation
    Config["general"]["sub_version"] = args.sub_version
    Config["general"]["WD"] = args.WD
    Config["general"]["blocks"] = args.blocks
    Config["general"]["SYMCH"] = args.SYMCH
    Config["general"]["level"] = args.level
    Config["general"]["LSize"] = args.LSize

    Config["general"]["DifuS_version"] = args.DifuS_version
    Config["general"]["Condition"] = args.Condition
    Config["general"]["ConditionCH"] = 1 if args.Condition == "01" else 2
    Config["general"]["SepFNODeCoder"] = args.SepFNODeCoder

    
    Config['LossWeightLddmm']['alpha'] = args.alpha
    Config['LossWeightLddmm']['gamma'] = args.gamma
    Config['LossWeightLddmm']['power'] = args.power

    Config["general"]["num_steps"] = args.num_steps
    Config["LStrategy"]["Registration"]["lr"] = args.LR

    Config["general"]["fixed_src"] = args.fixed_src

    Config["DataSet"]["dataName"] = args.dataName
    Config["DataSet"]["data_base_size"] = args.data_base_size
    Config["DataSet"]["train_type"] = args.train_type
    Config["DataSet"]["test_type"] = args.test_type

    Config["DataSet"]["Mnist_Tag"] = args.Mnist_Tag
    Config["DataSet"]["Shape_Tag"] = args.Shape_Tag
 
    Config["LStrategy"]["train_batch_size"] = args.train_batch_size
    Config["general"]["loadModelAtBegin"] = args.loadModelAtBegin
    #ModesFno
    Config["general"]["ModesFno"] = args.ModesFno
    #WidthFno
    Config["general"]["WidthFno"] = args.WidthFno
    #PlantType
    Config["DataSet"]["PlantType"] = args.PlantType
    #USE_AUG
    Config["DataSet"]["USE_AUG"] = args.USE_AUG
    #null_src_cond_prob null_cond_prob
    Config["general"]["null_src_cond_prob"] = args.null_src_cond_prob
    Config["general"]["null_cond_prob"] = args.null_cond_prob
    #cond_scale src_cond_scale
    Config["general"]["cond_scale"] = args.cond_scale
    Config["general"]["src_cond_scale"] = args.src_cond_scale
    




    module_name = Config["general"]["module_name"]  #JulyGDN_DifuS
    snapshot_path =  HOME+f'/data/project_complex/{Config["DataSet"]["train_type"]}_{Config["DataSet"]["train_img_size"]}/{Config["general"]["module_name"]}_lr{Config["LStrategy"]["Registration"]["lr"]}/'
    snapshot_path = snapshot_path + f'{Config["general"]["sub_version"]}'
    # print(Config["LStrategy"]["Registration"]["lr"])  #0.00011
    #max_epochs
    snapshot_path = snapshot_path + f'Ep{Config["LStrategy"]["max_epochs"]}'
    snapshot_path = snapshot_path + f'_dlb{Config["general"]["dim_linear_block"]}' if Config["general"]["dim_linear_block"] != 1024 else snapshot_path #dim_linear_block



    # snapshot_path = snapshot_path if Config["general"]["WD"] == 0.0001 else snapshot_path + f'_Ep{Config["LStrategy"]["max_epochs"]}'
    # advanced_config = load_config('advanced_config.json')
    # config = merge_configs(basic_config, advanced_config)
    # print(BaseConfig["LStrategy"]["batch_size"])

    
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(Config["general"]["seed"])
    np.random.seed(Config["general"]["seed"])
    torch.manual_seed(Config["general"]["seed"])
    torch.cuda.manual_seed(Config["general"]["seed"])

    
    
    
    if "ATT" in args.sub_version:
        snapshot_path = snapshot_path + f'_ATTblocks{args.blocks}'


    snapshot_path = snapshot_path + f'_bs{Config["LStrategy"]["train_batch_size"]}'
    snapshot_path = snapshot_path + f'_lrR{Config["LStrategy"]["Registration"]["lr"]}'
    snapshot_path = snapshot_path + f'_lrE{Config["LStrategy"]["Epdiff"]["lr"]}'
     # smoothness
    Config["LossWeightLddmm"]["alpha"] = args.alpha
    Config["LossWeightLddmm"]["gamma"] = args.gamma
    Config["LossWeightLddmm"]["power"] = args.power
    # print(args.alpha, args.gamma, args.power)   #1.0 0.5 2.0
    snapshot_path = snapshot_path + f'_lddmm-alpha-gamma-power-{Config["LossWeightLddmm"]["alpha"]}-{Config["LossWeightLddmm"]["gamma"]}-{Config["LossWeightLddmm"]["power"]}'

    #loss weights
    Config["LossWeightLddmm"]["WSimi"] = args.WSimi
    Config["LossWeightLddmm"]["sigma"] = args.sigma
    Config["LossWeightLddmm"]["WReg"] = args.WReg
    snapshot_path = snapshot_path + f'_loss-simi-sigma-reg-{Config["LossWeightLddmm"]["WSimi"]}-{Config["LossWeightLddmm"]["sigma"]}-{Config["LossWeightLddmm"]["WReg"]}'
   
    if "GDN" in args.module_name:
        Config["LossWeightLddmm"]["WEPDiff_Mse"] = args.WEPDiff_Mse
        Config["LossWeightLddmm"]["WEPDiff_Relative"] = args.WEPDiff_Relative
        snapshot_path = snapshot_path + f'_EPDiff-Mse-Relative-{Config["LossWeightLddmm"]["WEPDiff_Mse"]}-{Config["LossWeightLddmm"]["WEPDiff_Relative"]}'
    snapshot_path = snapshot_path + f'_nSteps{Config["general"]["num_steps"]}'

    snapshot_path = snapshot_path + f'_{args.DifuS_version}_Con_{args.Condition}' 
    snapshot_path = snapshot_path + f'_USE_AUG{args.USE_AUG}' if args.USE_AUG != "None" else snapshot_path
    snapshot_path = snapshot_path + f'_NSProb_{args.null_src_cond_prob}' if args.null_src_cond_prob != 0.0 else snapshot_path
    snapshot_path = snapshot_path + f'_NTProb_{args.null_cond_prob}' if args.null_cond_prob != 0.0 else snapshot_path

    dataName = Config["DataSet"]["dataName"] #OASIS32D
    print(Config["general"]["shootingType"]) #lddmm
    print(args.module_name, dataName)    #JulyGDN_DifuS
    print(snapshot_path)   # /home/nellie/data/project_complex/OASIS32D_128/JulyGDN_DifuS_lr0.00011/SATT101_UPS2Ep20000

    datapath = HOME+f'/{Config["DataPath"][dataName]}'
    #cond_scale src_cond_scale
    Config["general"]["snapshot_path"] = snapshot_path
    visjpg = snapshot_path + f'/view_res_{args.cond_scale}_{args.src_cond_scale}/{Config["DataSet"]["test_type"]}/{Config["DataSet"]["test_img_size"]}'
    Config["general"]["visjpg"] = visjpg

    netdic = {'JulyGDN_DifuS': NetJuly(Config).cuda()}
    trainer = {'JulyGDN_DifuS': trainer_GDN_DiFuS}
    tester = {'JulyGDN_DifuS': tester_GDN_DiFuS}
    net = netdic[module_name]
   

    if args.mode == "train":
        # if args.module_name == "JulyGDN_DifuS" and "Gray4" not in Config["DataSet"]["dataName"] and "OASIS" not in Config["DataSet"]["dataName"]:
        startIdx = Config["LStrategy"]["max_epochs"] - 1
        snapshot_path_ = snapshot_path.replace("JulyGDN_DifuS", "JulyGDN").replace("bs22_","bs20_").replace("bs36_","bs20_").replace("bs10_","bs20_").replace("bs4_","bs20_").replace("bs24_","bs20_").replace("bs16_","bs20_").replace("bs8_","bs20_").replace("bs5_","bs20_").replace("lr0.0001","lr0.0005").replace("lrR0.0001","lrR0.0005")[:-12]
        save_mode_path = os.path.join(snapshot_path_,   f'registration_{startIdx}.pth')
        while not os.path.exists(save_mode_path) and startIdx>=0:
            startIdx -= 1
            save_mode_path = os.path.join(snapshot_path_,   f'registration_{startIdx}.pth')
        
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        trainer[module_name](Config, net)


    elif args.mode == "test":
        if not os.path.exists(visjpg):
            os.makedirs(visjpg)
        temp = f"{visjpg}/temp"
        if not os.path.exists(temp):
            os.makedirs(temp)
        tester[module_name](Config, net)


if __name__ == "__main__":
    main()
