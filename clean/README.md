# Self-contained multiple expert models

## Generating Superresolution events

```bash
python generate.py --run-name ddim_v_multinode_64 --prediction-type v --sampler ddim   --base-width 64 --superres
```

## Evaluation with error metrics

```bash
python evaluate.py --run-name ddim_v_multinode_64 --prediction-type v --sampler ddim   --base-width 64  --Reynolds-number 16000 --batch-size 1
```

the additional flag ```--superres``` can be used instead to evaluate over superresolution.