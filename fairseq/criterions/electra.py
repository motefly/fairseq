# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('electra')
class ElectraLoss(FairseqCriterion):
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
        mask_idx = self.task.dictionary.index('<mask>')
        masked_tokens = sample['net_input']['src_tokens'].eq(mask_idx) # masked_tokens = sample['target'].ne(self.padding_idx)
        
        not_pad_tokens = sample['target'].ne(self.padding_idx)
        
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        gen_logits, disc_output, disc_tokens, _ = model(**sample['net_input'], masked_tokens=masked_tokens)
        targets = model.get_targets(sample, [gen_logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        gen_loss = F.nll_loss(
            F.log_softmax(
                gen_logits.view(-1, gen_logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        disc_targets = disc_tokens.eq(sample['target'])[not_pad_tokens].float()

        disc_loss = F.binary_cross_entropy_with_logits(disc_output[not_pad_tokens].float().view(-1),
            disc_targets.view(-1), reduction='sum')

        loss = gen_loss + self.args.loss_lamda * disc_loss

        logging_output = {
            'loss': utils.item(disc_loss.data) if reduce else disc_loss.data,
            'nll_loss': utils.item(gen_loss.data) if reduce else gen_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        logging_output.update(
            disc_loss=disc_loss.item()
        )
        logging_output.update(
            gen_loss=gen_loss.item()
        )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / ntokens / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        disc_loss = sum(log.get('disc_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(disc_loss=disc_loss)
        gen_loss = sum(log.get('gen_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(gen_loss=gen_loss)
        
        return agg_output
