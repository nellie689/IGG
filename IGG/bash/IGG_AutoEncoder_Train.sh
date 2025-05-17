#!/bin/bash
# 设置要调用的 Python 脚本路径和文件名
PYTHON_SCRIPT="../MainJuly.py"

# 在这里设置传递给 Python 脚本的参数（可选）
server="My"
mode="train"

#smoothness
shootingType="lddmm"
max_epochs=20000;blocks=1
LR=0.0005
ddN="PlantGray4"
# ddN="OASIS32D"



loadModelAtBegin="False"
if [ "$ddN" == "OASIS32D" ]; then
    train_type="OASIS32D"
    test_type="OASIS32D"
    dataName="OASIS32D"
    data_base_size=256
    module_name="JulyGDN";sub_version="SATT101_UPS2";SYMCH=1
    level=0;alpha=1.0;power=2;sigma=0.01;WSimi=0.5;device=2
    ## use this for fno-training
    SepFNODeCoder="Yes";
    # SepFNODeCoder="No"
    gamma=0.5;WReg=1.0;device=0
    ModesFno=8;WidthFno=20
    level=0;alpha=1.0;power=2;sigma=0.03;WSimi=0.5;device=2

elif [ "$ddN" == "PlantGray4" ]; then
    train_type="PlantGray4"
    test_type="PlantGray4"
    dataName="PlantGray4"
    data_base_size=204
    module_name="JulyGDN";sub_version="SATT101_UPS2";SYMCH=1
    level=0;alpha=1.0;power=2;sigma=0.01;WSimi=0.5;device=2

    ## use this for fno-training
    SepFNODeCoder="Yes";
    # SepFNODeCoder="No"
    gamma=0.5;WReg=1.0;device=0
    ModesFno=8;WidthFno=20


else
    echo "444 is not equal to desired_string"
fi



train_batch_size=12
WD=0
if [ $server = "My" ]; then
    device=0
fi
CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --mode $mode --max_epochs $max_epochs --server $server --module_name $module_name \
    --WSimi $WSimi --WReg $WReg --sub_version $sub_version --blocks $blocks --WD $WD --SYMCH $SYMCH --shootingType $shootingType \
    --alpha $alpha --power $power --gamma $gamma --sigma $sigma \
    --train_batch_size $train_batch_size --LR $LR \
    --dataName $dataName --data_base_size $data_base_size --train_type $train_type --test_type $test_type --level $level --SepFNODeCoder $SepFNODeCoder --loadModelAtBegin $loadModelAtBegin --ModesFno $ModesFno