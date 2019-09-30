TOTAL_UPDATES=100000    # Total number of training steps
WARMUP_UPDATES=1 #24000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=data-bin/wikitext-103
#python -m torch.distributed.launch --nproc_per_node=4 \
 #      --nnodes=2 --node_rank=1 --master_addr="10.0.10.4" \
  #     --master_port=8080 \
       $(which fairseq-train) --fp16 $DATA_DIR \
       --task masked_lm --criterion masked_lm \
       --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
       --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
       --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
       --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
       --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
       --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --save-interval-updates 10000 --keep-interval-updates 10\
       --encoder-normalize-before --skip-invalid-size-inputs-valid-test --tensorboard-logdir tsb_log # --cuda_ext 
     #   --distributed-world-size 16 --distributed-rank 12 --distributed-init-method "tcp://10.0.10.4:8080"\
       
       # --restore-file out_model/roberta_base/model.pt
