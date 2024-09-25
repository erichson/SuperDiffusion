CUDA_VISIBLE_DEVICES=0,1,2,3 python3 trainSR_nstk.py --run-name todo2 --sampling-fre 25 --epochs 250

# nohup bash -c "CUDA_VISIBLE_DEVICES=7 python3 trainSR_nstk.py" > output.log 2>&1 &
