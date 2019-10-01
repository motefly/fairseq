# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion
import numpy as np


@register_criterion('masked_lm')
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.vocab_num = len(task.dictionary)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None
            
        if model.training:
            # logits = model(**sample['net_input'], masked_tokens=masked_tokens)
            targets = model.get_targets(sample).cpu()
            items = np.copy(targets.view(-1))
            
            # print(items.shape, np.unique(items).shape)
            vocab_num = self.vocab_num
            # out = np.array([np.concatenate([np.array([items[i]]), np.random.randint(0, items[i], int(items[i]/vocab_num*511)), np.random.randint(items[i]+1, vocab_num, 511-int(items[i]/vocab_num*511))]) for i in range(len(items))])
            strange = np.setdiff1d(np.arange(vocab_num), items)
            hav = np.unique(items)
            sample_sz = min(max(2*hav.shape[0], 2048),8192) - hav.shape[0]
            out = np.concatenate([hav, np.random.choice(strange, sample_sz, replace=False)])
            tab = np.ones(vocab_num,dtype=np.int32)*-1
            for idx,item in enumerate(out):
                tab[item] = idx
            for idx,item in enumerate(items):
                items[idx] = tab[item]
            
            logits = model(**sample['net_input'], masked_tokens=torch.from_numpy(out).cuda().long())
        else:
            logits = model(**sample['net_input'], masked_tokens=None)
        if (isinstance(logits[0], tuple)):

            targets = torch.from_numpy(items).cuda().long() #model.get_targets(sample, [logits1])

            logits1 = logits[0][0]
            logits2 = logits[0][1]

            # print(logits1.shape)


            # if sample_size != 0:
            #     targets = targets[masked_tokens]

            # import pdb
            # pdb.set_trace()
            
            loss1 = F.nll_loss(
                F.log_softmax(
                    logits1.view(-1, logits1.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=int(tab[self.padding_idx]),
                )
            # loss = loss1
            loss2 = F.nll_loss(
                F.log_softmax(
                    logits2.view(-1, logits2.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=int(tab[self.padding_idx]),
            )
            loss = loss1 + loss2

        else:
            logits = logits[0]

            if sample_size != 0:
                targets = targets[masked_tokens]

            loss = F.nll_loss(
                F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
