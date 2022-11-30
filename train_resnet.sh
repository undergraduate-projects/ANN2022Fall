# check training data list for ade20k
training_data_list="ADE20K/datalist/training.txt"
if [ ! -f $training_data_list ]; then
    echo "Training data list not found!"
    echo "Try to generate it"
    python3 edit_dataset.py
    if [ ! -f $training_data_list ]; then
        echo "Failed to generate training data list!"
        exit 1
    fi
fi

# check pretrained model
jittor_pretrained_model="pretrain/resnet101_base.pth"
if [ ! -f $jittor_pretrained_model ]; then
    echo "Pretrained model not found!"
    echo "Try to convert it"
    python3 edit_pretrain_resnet.py
    if [ ! -f $jittor_pretrained_model ]; then
        echo "Failed to convert pretrained model!"
        exit 1
    fi
fi

# check log directory
if [ ! -d "log" ]; then
    mkdir "log"
fi
log_dir="log/resnet"
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

# name log file with date and time
log_file=$log_dir"/$(date +%Y%m%d_%H%M%S).log"

python3 train_resnet.py | tee $log_file