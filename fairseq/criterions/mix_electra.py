# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('mix_electra')
class MixElectraLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        with torch.autograd.profiler.record_function("forward"):
            mask_idx = self.task.dictionary.index('<mask>')

            masked_tokens = sample['net_input']['src_tokens'].eq(mask_idx) # 
            not_pad_tokens = sample['target'].ne(self.padding_idx)

            mlm_sample_size = (masked_tokens & not_pad_tokens).int().sum().item()
            bin_sample_size = ((~masked_tokens) & not_pad_tokens).int().sum().item()

            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if mlm_sample_size == 0:
                mlm_tokens = None
                bin_tokens = not_pad_tokens
            else:
                mlm_tokens = masked_tokens & not_pad_tokens
                bin_tokens = (~masked_tokens) & not_pad_tokens

            mask_logits, unmask_logits = model(**sample['net_input'], mlm_tokens=mlm_tokens, bin_tokens=bin_tokens)[0]
            targets = model.get_targets(sample, [mask_logits])

        if mlm_sample_size != 0:
            mask_targets = targets[mlm_tokens]
            unmask_targets = targets.eq(sample['net_input']['src_tokens'])[bin_tokens].float()
        with torch.autograd.profiler.record_function("loss"):
            loss1 = F.nll_loss(
                F.log_softmax(
                    mask_logits.view(-1, mask_logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                mask_targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            loss2 = F.binary_cross_entropy_with_logits(
                unmask_logits.float().view(-1),
                unmask_targets.view(-1),
                reduction='sum'
            )
            
            loss = loss1 + self.args.loss_lamda * loss2
        logging_output = {
            'loss': utils.item(loss2.data) if reduce else loss2.data,
            'nll_loss': utils.item(loss1.data) if reduce else loss1.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': mlm_sample_size,
            'bin_sample_size': bin_sample_size,
        }
        logging_output.update(
            mlm_loss=loss1.item()
        )
        logging_output.update(
            bin_loss=loss2.item()
        )
        return loss, mlm_sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        bin_sample_size = sum(log.get('bin_sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / bin_sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'bin_sample_size': bin_sample_size,
        }
        bin_loss = sum(log.get('bin_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(bin_loss=bin_loss)
        mlm_loss = sum(log.get('mlm_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(mlm_loss=mlm_loss)

        return agg_output
