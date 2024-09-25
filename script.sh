# CUDA_VISIBLE_DEVICES=6 python3 trainSR_nstk.py

nohup bash -c "CUDA_VISIBLE_DEVICES=7 python3 trainSR_nstk.py" > output.log 2>&1 &
