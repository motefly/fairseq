#!/usr/bin/env bash

EXEC_ID=replace-test-10-10-02
DATA_DIR=../data-bin/wiki_book_32768
TOTAL_UPDATES=1000000
WARMUP_UPDATES=10000
PEAK_LR=0.0005
TOKENS_PER_SAMPLE=128
MAX_POSITIONS=128
MAX_TOKENS=16384 # 32 for v100 and fp16
UPDATE_FREQ=1
SEED=100

# echo 'Environment'
# nvidia-smi
# ls -alh
# ls ~ -alh

# echo 'Prepare Code'
# git clone https://github.com/motefly/fairseq.git
# cd fairseq
# git checkout mix_electra
# sudo -H /opt/conda/envs/pytorch-py3.7/bin/python -m pip install --editable .

# echo 'Prepare Data'
# mkdir -p data-bin
# cp -r /blob/v-zhexu/${DATA_DIR} data-bin/

echo 'Start Training'
python train.py ${DATA_DIR} --ddp-backend=no_c10d \
       --task mix_electra --criterion mix_electra \
       --arch mixelectra_small --sample-break-mode complete_doc --tokens-per-sample ${TOKENS_PER_SAMPLE} \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
       --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
       --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
       --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} --seed ${SEED} \
       --loss-lamda 2.0 --mask-prob 0.10 --random-replace-prob 0.10 --encoder-normalize-before  \
       --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 100 --tensorboard-logdir ../output_tsb/mix_electra-${EXEC_ID} \
       --keep-updates-list 20000 50000 100000 200000 400000 600000 800000 1000000 \
       --save-interval-updates 25000 --keep-interval-updates 5 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test --save-dir ../output_cp/mix_electra-${EXEC_ID}
