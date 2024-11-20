#!/bin/bash
#SBATCH -p a5
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-[5,6]
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="mast_as"
#SBATCH --output=./log_%x.%j.txt

set -x
# comment this line if not running on sls cluster
CONDA_ROOT=/data/sls/scratch/sbhati/miniconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate DASS
which python
export TORCH_HOME=../../pretrained_models

model=$1
model_size=$2
dataset=audioset
# full or balanced for audioset
set=$3
imagenetpretrain=$4
knowledge_distillation=$5
dist_loss_type=$6
dist_temp=1.0
if [ $set == balanced ]
then
  bal=none
  lr=$7
  epoch=25
  tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/balanced_train_data_type1_2_mean.json
  lrscheduler_start=10
  lrscheduler_step=5
  lrscheduler_decay=0.5
  wa_start=6
  wa_end=25
else
  bal=bal
  lr=$7
  epoch=5
  tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
  lrscheduler_start=2
  lrscheduler_step=1
  lrscheduler_decay=0.5
  wa_start=1
  wa_end=5
fi
te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=12

dataset_mean=-4.2677393
dataset_std=4.5689974
audio_length=1024
noise=False

metrics=mAP
loss=BCE
warmup=True
wa=True

if [ $model == DASS ]
then
exp_dir=./exp/${model}-${model_size}-${set}-p$imagenetpretrain-b$batch_size-lr${lr}-kd${knowledge_distillation}-kdd${dist_loss_type}-dt${dist_temp}
fi

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --model_size ${model_size} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
--knowledge_distillation ${knowledge_distillation} --dist_loss_type ${dist_loss_type} --dist_temp ${dist_temp} 