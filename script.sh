### SR task
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 trainSR_nstk.py --run-name todo2 --sampling-fre 25 --epochs 250

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 trainSR_nstk.py --run-name full0929cont --sampling-fre 10 --epochs 100

# nohup bash -c "CUDA_VISIBLE_DEVICES=7 python3 trainSR_nstk.py" > output.log 2>&1 &

### FC task
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 trainFC_nstk.py --run-name full1022interactive --sampling-fre 20 --epochs 61 --batch-size 1024
# CUDA_VISIBLE_DEVICES=0,1,2,3