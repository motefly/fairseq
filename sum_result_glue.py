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
parser.add_argument('-p2s', type=str, nargs='+', required = True)
parser.add_argument('-cs',type=str, nargs='+', required = True)
# parser.add_argument('-v',type=str, required = True, choices=['v1', 'v2'])
parser.add_argument('-arch',type=str, required = True)
parser.add_argument('-task_names',type=str, nargs='+', required = True)

args = parser.parse_args()


project_folder = "electra" # "adaptive_bert"  # !! TO UPDATE

def readResult(path, keyword='best_'):
    files = os.listdir(path)
    s = []
    done_mark = "done training"
    for File in files:
        ok = False
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
                if done_mark in line:
                    ok = True
            if not ok:
                print(path, "not ok!")
        cols = File.split('-')[:-1]
        # for idx,col in enumerate(cols):
        #     summ[idx].append(col)
        cols.append(ans)
        summ.append(cols)
    return pd.DataFrame(summ)
ress = {}
for p2 in args.p2s:
    anss = []
    for task in args.task_names:
        ans = []
        for c in args.cs:
            bert_model_config = {
                "bert_model_arch": args.arch,
                "bert_model_checkpoint": "checkpoint{}".format(c),
                "procedure_folder1": args.p1,
                "procedure_folder2": p2,
            }  
            bert_model_config["procedure_path"] = "{}/{}".format(bert_model_config["procedure_folder1"], bert_model_config["procedure_folder2"])
            
            logpath="big-{}/GLUE/{}/{}".format(bert_model_config["procedure_path"], bert_model_config["bert_model_checkpoint"], task)
        
            summ = []
            
            results = readResult(logpath)
            # results.groupby(['0','1','2','3'])
            results = results.join(results.groupby([0,1,2,3])[5].mean(), on=[0,1,2,3], rsuffix='_mean')
            results = results.join(results.groupby([0,1,2,3])['5'].std(), on=[0,1,2,3], rsuffix='_std')
            results = results.sort_values(by='5_mean')
            print(p2,task,c,results.shape,results.values[-1])
            result = results.values[-1][6]
            ans.append(result)
        anss.append(ans)
    ress[p2]= anss
print(ress)
        # print(results.sort_values(by='5_mean')[-20:])
        
        # results.to_csv(logpath.replace('/','_')+'_result.csv', index=False)
        # print("file saved at {}".format(logpath.replace('/','_')+'_result.csv'))
    
