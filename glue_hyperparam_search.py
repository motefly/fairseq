import os
import sys
import copy
import itertools
import inspect



import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--philly', action = 'store_true')
parser.add_argument('-p1', type=str, default = '/gpu-01-data/zhenhui/downstream') 
parser.add_argument('-p2', type=str, required = True)
parser.add_argument('-c',type=str, required = True)
# parser.add_argument('-v',type=str, required = True, choices=['v1', 'v2'])
parser.add_argument('-arch',type=str, required = True)

args = parser.parse_args()


project_folder = "mix_electra" # "adaptive_bert"  # !! TO UPDATE

bert_model_config = {
    "bert_model_arch": args.arch, #"transformer_v1_classifier_base" if args.v == 'v1' else "transformer_classifier_base",
    "bert_model_checkpoint": "checkpoint_{}".format(args.c),
    "procedure_folder1": args.p1,
    "procedure_folder2": args.p2,
}  
bert_model_config["procedure_path"] = "{}/{}".format(bert_model_config["procedure_folder1"], bert_model_config["procedure_folder2"])



def task(name, n_sentences, metric, criterion, symmetric, n_classes, data_path):
    return locals()

def params(*args):
    keys = ["seed_list", "n_epoch_list", "batch_sz_list", "lr_list", "weight_decay_list"]
    assert len(args) == len(keys)
    values = itertools.product(*args)
    return [{k: v for k, v in zip(keys, vs)} for vs in values]

# cola = (
#     task("CoLA", 8551, "mcc", "cross_entropy_classify_binary", "", 2, "CoLA"),
#     # params(["100 200 300 400 500 600"], ["5 6 7 8"], ["16 32"], ["0.00005 0.0006", "0.00007 0.00008", "0.00009 0.00010", "0.000011 0.00012"], ["0.01"])
#     params(["100 200 300"], ["10"], ["16", "32"], ["0.00003 0.00004", "0.00005 0.00006"], ["0.01"])
# ) # 60s / epoch, 3h / search
mrpc = (
    task("MRPC", 3668, "acc_f1", "cross_entropy_classify_binary", "--symmetric", 2, "MRPC"), # change original num_class 1 to 2
    # params(["100 200 300 400 500 600"], ["3 4 5 6 7 8 9 10"], ["16", "32"], ["0.00003 0.00004 0.00005 0.00006", "0.00007 0.00008 0.00009 0.0001"], ["0.00 0.01"]) # original
    # params(["100 200 300", "400 500 600"], ["5 6 7 8 9 10"], ["16 32"], ["0.00003 0.00004 0.00005 0.00006", "0.00007 0.00008 0.00009 0.0001"], ["0.00 0.01"]) # for roberta
    params(["100 200 300", "400 500 600"], ["10"], ["16 32"], ["0.00003 0.00004 0.00005 0.00006", "0.00007 0.00008 0.00009 0.0001"], ["0.1"])
) # 50s / epoch, 3h / search
# sts = (
#     task("STS-B", 5749, "glue_pair", "mean_squared_error", "--symmetric", 1, "STS-B"),
#     params(["100 200 300 400 500 600"], ["3 4 5 8"], ["16 32"], ["0.00005 0.00003 0.00002"], ["0.01"])
# ) # 50s / epoch, 4h / search
# rte = (
#     task("RTE", 2475, "accuracy", "cross_entropy_classify", "", 2, "RTE"),
#     # params(["100 200 300 400 500 600"], ["3 4 5 6 7 8 9 10"], ["16", "32"], ["0.00003 0.00004 0.00005 0.00006", "0.00007 0.00008 0.00009 0.0001"], ["0.00 0.01"]), #original
#     # params(["100 200 300 400 500 600"], ["6 7 8 9 10"], ["16"], ["0.00003 0.00004 0.00005 0.00006", "0.00007 0.00008 0.00009 0.0001"], ["0.00", "0.01"]) # for roberta
#     params(["100 200 300"], ["10"], ["16", "32"], ["0.00003 0.00004", "0.00005 0.00006"], ["0.01"])

# ) # 60s / epoch, 3h / search
# mnli = (
#     task("MNLI", 392702, "glue_pair", "cross_entropy_classify", "", 3, "MNLI"),
#     params(["100", "200", "300"], ["3 4 5"], ["16 24"], ["0.00005", "0.00003", "0.00002"], ["0.01"])
# ) # 5000s / epoch, bs 32 oom
# mnlimm = (
#     task("MNLI-mm", 392702, "glue_pair", "cross_entropy_classify", "", 3, "MNLI-mm"),
#     params(["100", "200", "300"], ["3 4 5"], ["16 24"], ["0.00005", "0.00003", "0.00002"], ["0.01"])
# ) # 5000s / epoch, bs 32 oom
# qnli = (
#     task("QNLI", 108436, "glue_pair", "cross_entropy_classify", "", 2, "QNLI"),
#     params(["100", "200", "300"], ["3 4 5"], ["16 24"], ["0.00005", "0.00003", "0.00002"], ["0.01"])
# ) # 1600s / epoch, bs 32 oom
# qqp = (
#     task("QQP", 363849, "glue_pair", "cross_entropy_classify_binary", "--symmetric", 1, "QQP"),
#     params(["100", "200", "300"], ["3 4 5"], ["16", "24"], ["0.00005", "0.00003", "0.00002"], ["0.01"])
# ) # 4000s / epoch, bs 32 oom
sst = (
    task("SST-2", 67349, "accuracy", "cross_entropy_classify", "", 2, "SST-2"),
    params(["100 200 300"], ["10"], ["16 32"], ["0.00001", "0.00002", "0.00003", "0.00004"], ["0.1"])
) # 400s / epoch, 18h / search


task_list = [mrpc, sst]
script_dir = os.path.join("submit/glue/finetune/{}/{}".format(bert_model_config["procedure_folder2"], bert_model_config["bert_model_checkpoint"]))

env_vars = """
PROBLEM={name}
METRIC={metric}
CHECKPOINT_FILE={bert_model_checkpoint}
PROCEDURE_FOLDER={procedure_path}
BERT_MODEL_PATH={procedure_path}/{bert_model_checkpoint}.pt

N_CLASSES={n_classes}
ARCH={bert_model_arch}
N_SENT={n_sentences}
CRITERION={criterion}
SYMMETRIC={symmetric}
SEED_LIST="{seed_list}" 
N_EPOCH_LIST="{n_epoch_list}" 
BATCH_SZ_LIST="{batch_sz_list}" 
LR_LIST="{lr_list}" 
WEIGHT_DECAY_LIST="{weight_decay_list}" 

CODE_HOME=/home/zhenhui/my-fairseq/{project_folder}
DATA_PATH=/home/zhenhui/my-fairseq/data-ds/glue-32768-fast
TENSORBOARD_LOG={procedure_path}/GLUE/{bert_model_checkpoint}
"""


script_template = r"""
cd $CODE_HOME

for SEED in $SEED_LIST
do
    for N_EPOCH in $N_EPOCH_LIST
    do
        for BATCH_SZ in $BATCH_SZ_LIST
        do
            SENT_PER_GPU=$(( BATCH_SZ / 1 ))
            N_UPDATES=$(( ((N_SENT + BATCH_SZ - 1) / BATCH_SZ) * N_EPOCH ))
            WARMUP_UPDATES=$(( N_UPDATES / 16 ))
            echo $SENT_PER_GPU $N_UPDATES $WARMUP_UPDATES
            for LR in $LR_LIST
            do
                for WEIGHT_DECAY in $WEIGHT_DECAY_LIST
                do

echo ${N_EPOCH} ${BATCH_SZ} ${LR} ${WEIGHT_DECAY} $SEED
OUTPUT_PATH=big-${PROCEDURE_FOLDER}/GLUE/${CHECKPOINT_FILE}/${PROBLEM}/${N_EPOCH}-${BATCH_SZ}-${LR}-${WEIGHT_DECAY}-$SEED-0.006
mkdir -p $OUTPUT_PATH

if [ -e $OUTPUT_PATH/ok ]; then 
    continue
fi

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
FILE_NAME=$(basename $0)
cd $CODE_HOME
cp ${SHELL_FOLDER}/${FILE_NAME} ${OUTPUT_PATH}/${FILE_NAME}

python train.py $DATA_PATH/${PROBLEM}-bin \
       --restore-file $BERT_MODEL_PATH \
       --max-positions 128 \
       --max-sentences $SENT_PER_GPU \
       --max-tokens 4400 \
       --task sentence_prediction \
       --reset-optimizer --reset-dataloader --reset-meters \
       --required-batch-size-multiple 1 \
       --init-token 0 --separator-token 2 \
       --arch $ARCH \
       --criterion sentence_prediction \
       --num-classes $N_CLASSES \
       --dropout 0.1 --attention-dropout 0.1 \
       --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
       --clip-norm 0.0 \
       --lr-scheduler polynomial_decay --lr $LR --total-num-update $N_UPDATES --warmup-updates $WARMUP_UPDATES\
       --max-epoch $N_EPOCH --seed $SEED --save-dir $OUTPUT_PATH --no-progress-bar --log-interval 100 --no-epoch-checkpoints \
       --find-unused-parameters --skip-invalid-size-inputs-valid-test --encoder-normalize-before --truncate-sequence \
       --tensorboard-logdir $TENSORBOARD_LOG/${PROBLEM}/${N_EPOCH}-${BATCH_SZ}-${LR}-${WEIGHT_DECAY}-$SEED-0.006 \
       --best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric | tee -a $OUTPUT_PATH/train_log.txt

touch $OUTPUT_PATH/ok
rm -rf $OUTPUT_PATH/checkpoint_best.pt
rm -rf $OUTPUT_PATH/checkpoint_last.pt

                done
            done
        done
    done
done
"""

os.makedirs(script_dir, exist_ok=True)
os.system('cp {} {}'.format(__file__, script_dir))

scripts = []
from collections import defaultdict
scripts_dict = defaultdict(list)

count = 0
for task_dict, params_list in task_list:
    for i, param_dict in enumerate(params_list):
        result_dict = {}
        result_dict['project_folder'] = project_folder
        result_dict.update(task_dict)
        result_dict.update(bert_model_config)
        result_dict.update(param_dict)
        this_env_var = env_vars.format(**result_dict)
        script = this_env_var + script_template
        script_name = os.path.join(script_dir, ".".join([task_dict["name"], "%02d" % i, "sh"]))
        scripts.append(script_name)
        print(script_name)
        scripts_dict[task_dict["name"]].append(script_name)
        with open(script_name, "w") as f:
            f.write(script)
        count += 1

print(count)
# content = "cd /blob/v-ruxion/code/{}\n".format(project_folder)


# mask_name = {
#     'CoLA': 'c',
#     'MRPC': 'mr',
#     "STS-B": 'st',
#     'RTE':'r',
#     'MNLI-mm':'mm',
#     "MNLI":'mn',
#     "QNLI":'qn',
#     "QQP":'q',
#     "SST-2":'ss'
# }



# if args.philly:
#     import subprocess
#     # Use Azcopy 
#     os.chdir('/home/v-ruxion/')
#     subprocess.call('bash copy-adaptive-bert.sh', shell = True)
#     # Use phiily submit
#     for task, scripts in scripts_dict.items():
#         for i, script in enumerate(scripts):
#             configfile = '{}/{}'.format(project_folder,script)
#             subprocess.call('python bin/philly_submit.py --configfile {} --jobname f-g-b-{}-{}-{}-{} --gpus 1'.format(configfile, args.v[-1], mask_name[task], i, args.c), shell=True)



