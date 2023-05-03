This is the repo our paper: Towards Being Parameter-Efficient: A Stratified Sparsely Activated Transformer with Dynamic Capacity

## Building VirtualEnvironments:
```
conda create -n fairseq-smoe python=3.8
conda activate fairseq-smoe
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -e ./
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout origin/experts_lt_gpus_moe_reload_fix
pip install .
```

## Download m15 dataset:
```
pip install gdown
gdown https://drive.google.com/uc?id=1tT2urUunXLNEPZxxV2nK-YmC_N6P_48m
unzip m15.zip
```
## Training SMoE:
The training command is in the following format (taking running experiments on M4 dataset):
```
sbatch (bash) runs/run_m4.sh ${SAVE_PATH} ${MOE_TYPE} ${TOTAL_EXPERT_NUM} ${HMOE_LAYER} ${EXPERT_NUM_PER_LAYER}
```
The explanation of these options are:
* `${SAVE_PATH}`: the place you save for the model checkpoints
* `${MOE_TYPE}`: we only supports `base`, `eom` and `cmr`.
    * `base`: the naive MoE model method
    * `eom`: MoE with EoM method, which can be used with SMoE together
    * `cmr`: MoE with CMR method, which can NOT be used with SMoE together
* `${TOTAL_EXPERT_NUM}` The total expert number
* `${HMOE_LAYER}`: The number of layers for SMoE
* `${EXPERT_NUM_PER_LAYER}`: The number of experts per layer, e.g., if you have 3 layers, you want 4 layers in the 1st layer, 4 layers in the 2nd layer and 8 layers in the 3rd layer, it should be `4,4,8`. The number is splited by a comma. Please ensure that 4+4+8=`${TOTAL_EXPERT_NUM}`.

Example: if we want to train a 4 layer SMoE, each layer has 2 expert based on a naive MoE architecture, and store the models to `./tmp` folder.
```
sbatch (bash) runs/run_m4.sh tmp base 8 4 2,2,2,2
```
The same training command also applies to M15 dataset by using `runs/run_m15.sh` file. Note that one should always enable the `--only-compute-current-layer-tokens` to reduce the GPU memory (which is already enabled in the training command). Using this option can make expert only compute tokens in its current strata.

## Evaluation:
The evaluation commands are below the training commands in the `runs/run_m{4,15}.sh` files. Comment the training commands and rerun them to conduct evaluation.
