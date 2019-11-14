import sys

import torch


def main():
    ckpt = torch.load(sys.argv[1])

    lst = []
    for k, v in ckpt['model'].items():
        k_split = k.split('.')
        if k_split[0] == 'decoder' and k_split[1] == 'sentence_encoder' and k_split[2] == 'layers':
            l_id = int(k_split[3])
            k_split[3] = str(l_id + ckpt['args'].encoder_layers)
            new_k = '.'.join(k_split)
            lst.append([new_k, v.clone()])
    for k, v in lst:
        ckpt['model'][k] = v

    ckpt['args'].encoder_layers *= 2
    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()
