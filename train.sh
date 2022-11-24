# check log directory
if [ ! -d "log" ]; then
    mkdir log
fi

# name log file with date and time
log_file="log/$(date +%Y%m%d_%H%M%S).log"

python3 train.py > $log_file 2>&1