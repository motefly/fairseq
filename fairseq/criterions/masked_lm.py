# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('masked_lm')
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        # self.args = args

    def forward(self, model, sample, reduce=True, lamda=0.5):
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

        # import pdb
        # pdb.set_trace()

        # logits = model(**sample['net_input'], masked_tokens=masked_tokens)
        logits = model(**sample['net_input'], masked_tokens=None)
        if(isinstance(logits[0], tuple)):
            logits1 = logits[0][0]
            logits2 = logits[0][1]

            targets = model.get_targets(sample, [logits1])

            # if sample_size != 0:
            #     targets = targets[masked_tokens]

            loss1 = F.nll_loss(
                F.log_softmax(
                    logits1.view(-1, logits1.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            loss2 = F.nll_loss(
                F.log_softmax(
                    logits2.view(-1, logits2.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            loss = lamda * loss1 + (1-lamda) * loss2

        else:
            logits = logits[0]
            targets = model.get_targets(sample, [logits])

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
