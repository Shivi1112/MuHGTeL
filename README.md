# MuHGTeL
MuHGTeL: A Multilayered Heterogeneous Graph Framework for Temporal-Semantic Alignment in EHR-based Disease Trajectory Modeling

## Requirements

Requirements and recommended versions:

Python (3.10.13)

pytorch (1.12.1)

torch-geometric (2.3.1)

Pyhealth (1.1.4)

## Data Processing

For MIMIC-III and MIMIC-IV: refer to https://pyhealth.readthedocs.io/en/latest/api/datasets.html; 


## Training & Evaluation

To train the model and baselines in the paper, run this command:

```
python train.py --model <MuHGTeL/Transformer/...> --dataset <mimic3/mimic4>
```
