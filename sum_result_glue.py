import os
import sys
import copy
import itertools
import inspect
import pandas as pd



import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--philly', action = 'store_true')
parser.add_argument('-p1', type=str, default = '/gpu-01-data/zhenhui/downstream') 
parser.add_argument('-p2', type=str, required = True)
parser.add_argument('-c',type=str, required = True)
# parser.add_argument('-v',type=str, required = True, choices=['v1', 'v2'])
parser.add_argument('-arch',type=str, required = True)
parser.add_argument('-task_name',type=str, required = True)

args = parser.parse_args()


project_folder = "electra" # "adaptive_bert"  # !! TO UPDATE


bert_model_config = {
    "bert_model_arch": args.arch,
    "bert_model_checkpoint": "checkpoint{}".format(args.c),
    "procedure_folder1": args.p1,
    "procedure_folder2": args.p2,
}  
bert_model_config["procedure_path"] = "{}/{}".format(bert_model_config["procedure_folder1"], bert_model_config["procedure_folder2"])

logpath="big-{}/GLUE/{}/{}".format(bert_model_config["procedure_path"], bert_model_config["bert_model_checkpoint"], args.task_name)

summ = []

def readResult(path, keyword='best_'):
    files = os.listdir(path)
    s = []
    for File in files:
        ans = 0
        with open(path+'/'+File+'/'+'train_log.txt', 'r') as log:
            for line in log.readlines():
                if keyword in line:
                    items = line.split(' ')
                    pos = -1
                    try:
                        for idx,item in enumerate(items):
                            if keyword in item:
                                pos = idx
                                break
                        if pos >= 0:
                            ans = max(ans, eval(items[pos+1]))
                    except:
                        continue
        cols = File.split('-')[:-1]
        # for idx,col in enumerate(cols):
        #     summ[idx].append(col)
        cols.append(ans)
        summ.append(cols)
    return pd.DataFrame(summ)

results = readResult(logpath)
import pdb
pdb.set_trace()
# results.groupby(['0','1','2','3'])
results = results.join(results.groupby([0,1,2,3])[5].mean(), on=[0,1,2,3], rsuffix='_mean')
results = results.join(results.groupby([0,1,2,3])['5'].std(), on=[0,1,2,3], rsuffix='_std')
print(results.sort_values(by='5_mean')[-20:])

results.to_csv(logpath.replace('/','_')+'_result.csv', index=False)
print("file saved at {}".format(logpath.replace('/','_')+'_result.csv'))
