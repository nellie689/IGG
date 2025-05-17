#!/bin/bash
# 设置要调用的 Python 脚本路径和文件名
PYTHON_SCRIPT="../MainJuly.py"
# 在这里设置传递给 Python 脚本的参数（可选）
server="My"
Shape_Tag="100";Mnist_Tag=100
max_epochs=20000;blocks=1
shootingType="lddmm"
LR=0.0001;alpha=1.0;power=2;gamma=0.5  # 
# LR=0.00012  # No text guidance
LR=0.00011  # Has text guidance

## fixed_src 只在test时用
# fixed_src!=-1  #计算confidence map: CDM
# fixed_src=-1  #计算velocity distribution and regularity
ddN="PlantGray4";PlantType="4&8&3Size";USE_AUG="Yes"
# PlantType="AsNivetha"
# ddN="OASIS32D"

null_cond_prob=0.0
null_src_cond_prob=0.2
# null_src_cond_prob=0.5
# null_src_cond_prob=0.7

null_cond_prob=999
null_src_cond_prob=999

null_cond_prob=0.5
null_src_cond_prob=0.5

# null_cond_prob=0.05
# null_src_cond_prob=0.05
mode="train"
mode="test"



if [ "$ddN" == "PlantGray4" ]; then
    train_type="PlantGray4"
    test_type="PlantGray4"
    dataName="PlantGray4"
    data_base_size=204
    # # # #Plant  
    module_name="JulyGDN_DifuS";sub_version="SATT101_UPS2";SYMCH=1;Condition="02"
    DifuS_version="way1";train_batch_size=12;device=0
    WReg=1.0;WSimi=0.5;sigma=0.01
    
    sub_version="SATT101_UPS2";SYMCH=1
    level=0
    SepFNODeCoder="Yes";
    gamma=0.5;WReg=1.0;device=0
    ModesFno=8;WidthFno=20

    
elif [ "$ddN" == "OASIS32D" ]; then
    train_type="OASIS32D"
    test_type="OASIS32D"
    dataName="OASIS32D"
    data_base_size=256
    DifuS_version="way1";train_batch_size=22
    # # # #OASIS32D  
    module_name="JulyGDN_DifuS";sub_version="SATT101_UPS2";SYMCH=1;Condition="02"
    DifuS_version="way1";train_batch_size=12;device=0
    WReg=1.0;WSimi=0.5;sigma=0.01
    
    sub_version="SATT101_UPS2";SYMCH=1
    level=0
    SepFNODeCoder="Yes";
    gamma=0.5;WReg=1.0;device=0
    ModesFno=8;WidthFno=20

    sigma=0.03;device=1


else
    echo "444 is not equal to desired_string"
fi



if [ "$server" = "My" ]; then
    device=0
elif [ "$server" = "Ri" ]; then
    device=0
else
    device=-1
fi

#plants
# comfidence map
fixed_src=-1
fixed_src=26
# fixed_src=47
# fixed_src=48
# fixed_src=61
# fixed_src=46
# fixed_src=17
# show examples
fixed_src=26
# fixed_src=17
# fixed_src=48
# fixed_src=46


# Brain
# fixed_src=-1
# fixed_src=4        #0
# # fixed_src=5      #1
# # fixed_src=72     #2
# # fixed_src=88      #4
# # fixed_src=120   #6
# # fixed_src=127   #7


fixed_src=75      #3
fixed_src=93      #5



# Plants
##fixed_src=24
## fixed_src=25
## fixed_src=26
## fixed_src=27

# fixed_src=28

fixed_src=47
fixed_src=49
fixed_src=60
fixed_src=61
fixed_src=17


cond_scale=2.0
src_cond_scale=3.0


cond_scale=1.0
src_cond_scale=1.0

# cond_scale=1.5
# src_cond_scale=1.5

# cond_scale=0.8
# src_cond_scale=0.8





fixed_src=-1
fixed_src=21
fixed_src=27

fixed_src=57
fixed_src=59
fixed_src=61

train_batch_size=8
# train_batch_size=40  # for AsNivetha
# fixed_src=-1
# fixed_src=56  #plants
fixed_src=58

train_batch_size=48   #brain
fixed_src=59
# fixed_src=56 #oasis32d
# fixed_src=59 #oasis32d
CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --mode $mode --max_epochs $max_epochs --server $server --module_name $module_name \
    --WSimi $WSimi --WReg $WReg --sub_version $sub_version --blocks $blocks --SYMCH $SYMCH --shootingType $shootingType \
    --alpha $alpha --power $power --gamma $gamma  --sigma $sigma --SepFNODeCoder $SepFNODeCoder --ModesFno $ModesFno --WidthFno $WidthFno \
    --train_batch_size $train_batch_size --LR $LR --DifuS_version $DifuS_version --Condition $Condition --fixed_src $fixed_src --USE_AUG $USE_AUG \
    --dataName $dataName --data_base_size $data_base_size --train_type $train_type --test_type $test_type --Mnist_Tag $Mnist_Tag --Shape_Tag $Shape_Tag --PlantType $PlantType \
    --null_src_cond_prob $null_src_cond_prob --null_cond_prob $null_cond_prob --cond_scale $cond_scale --src_cond_scale $src_cond_scale

# fixed_src=-1  #计算velocity distribution and regularity
# CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --mode $mode --max_epochs $max_epochs --server $server --module_name $module_name \
#     --WSimi $WSimi --WReg $WReg --sub_version $sub_version --blocks $blocks --SYMCH $SYMCH --shootingType $shootingType \
#     --alpha $alpha --power $power --gamma $gamma  --sigma $sigma --SepFNODeCoder $SepFNODeCoder --ModesFno $ModesFno --WidthFno $WidthFno \
#     --train_batch_size $train_batch_size --LR $LR --DifuS_version $DifuS_version --Condition $Condition --fixed_src $fixed_src --USE_AUG $USE_AUG \
#     --dataName $dataName --data_base_size $data_base_size --train_type $train_type --test_type $test_type --Mnist_Tag $Mnist_Tag --Shape_Tag $Shape_Tag --PlantType $PlantType




# # 遍历 0 到 61 的值   seq 0 61   seq 0 137
# for fixed_src in $(seq 100 137); do
#     echo "Running script with fixed_src=$fixed_src"
#     CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --mode $mode --max_epochs $max_epochs --server $server --module_name $module_name \
#         --WSimi $WSimi --WReg $WReg --sub_version $sub_version --blocks $blocks --SYMCH $SYMCH --shootingType $shootingType \
#         --alpha $alpha --power $power --gamma $gamma  --sigma $sigma --SepFNODeCoder $SepFNODeCoder --ModesFno $ModesFno --WidthFno $WidthFno \
#         --train_batch_size $train_batch_size --LR $LR --DifuS_version $DifuS_version --Condition $Condition --fixed_src $fixed_src --USE_AUG $USE_AUG \
#         --dataName $dataName --data_base_size $data_base_size --train_type $train_type --test_type $test_type --Mnist_Tag $Mnist_Tag --Shape_Tag $Shape_Tag --PlantType $PlantType \
#         --null_src_cond_prob $null_src_cond_prob --null_cond_prob $null_cond_prob --cond_scale $cond_scale --src_cond_scale $src_cond_scale
# done
