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
        mask_idx = self.task.dictionary.index('<mask>')

        masked_tokens = sample['net_input']['src_tokens'].eq(mask_idx) # 
        not_pad_tokens = sample['target'].ne(self.padding_idx)

        mlm_sample_size = (masked_tokens & not_pad_tokens).int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if mlm_sample_size == 0:
            mlm_tokens = None
            bin_tokens = not_pad_tokens
        else:
            mlm_tokens = masked_tokens & not_pad_tokens
            bin_tokens = (~masked_tokens) & not_pad_tokens

        bin_sample_size = bin_tokens.int().sum().item()

        mask_logits, unmask_logits = model(**sample['net_input'], mlm_tokens=mlm_tokens, bin_tokens=bin_tokens)[0]
        targets = model.get_targets(sample, [mask_logits])
        unmask_targets = targets.eq(sample['net_input']['src_tokens'])[bin_tokens].float()
        
        if self.args.random_replace_prob > 0.0:
            loss2 = F.binary_cross_entropy_with_logits(
                unmask_logits.float().view(-1),
                unmask_targets.view(-1),
                reduction='sum'
            )
            tp = ((unmask_logits.float().view(-1) >= 0) & (unmask_targets == 1)).long().sum()
            fp = ((unmask_logits.float().view(-1) >= 0) & (unmask_targets == 0)).long().sum()
            fn = ((unmask_logits.float().view(-1) < 0) & (unmask_targets == 1)).long().sum()
            tn = ((unmask_logits.float().view(-1) < 0) & (unmask_targets == 0)).long().sum()
            assert (tp + fp + tn + fn) == unmask_targets.size(0), 'invalid size'
        else:
            loss2 = torch.tensor(0.0)
        
        if mlm_sample_size != 0:
            mask_targets = targets[mlm_tokens]

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
            loss = loss1 + self.args.loss_lamda * loss2 * mlm_sample_size / bin_sample_size
        else:
            loss1 = torch.tensor(0.0)
            loss = self.args.loss_lamda * loss2 / bin_sample_size

        mlm_sample_size = mlm_sample_size if mlm_sample_size!=0 else 1
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
            bin_loss=loss2.item()*mlm_sample_size / bin_sample_size
        )
        if self.args.random_replace_prob > 0.0:
            logging_output.update(tp = utils.item(tp.data) if reduce else tp.data)
            logging_output.update(fp = utils.item(fp.data) if reduce else fp.data)
            logging_output.update(fn = utils.item(fn.data) if reduce else fn.data)
            logging_output.update(tn = utils.item(tn.data) if reduce else tn.data)

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

        if 'tp' in logging_outputs[0]: 
            tp_sum = sum(log.get('tp', 0) for log in logging_outputs)
            fp_sum = sum(log.get('fp', 0) for log in logging_outputs)
            fn_sum = sum(log.get('fn', 0) for log in logging_outputs)
            tn_sum = sum(log.get('tn', 0) for log in logging_outputs)
            assert tp_sum + fp_sum + fn_sum + tn_sum == bin_sample_size, 'invalid size when aggregating'
            bin_acc = (tp_sum + tn_sum) / bin_sample_size
            replace_acc = tn_sum / (tn_sum + fp_sum + 1e-5)
            non_replace_acc = tp_sum / (tp_sum + fn_sum + 1e-5)
            agg_output.update(bin_acc=bin_acc)
            agg_output.update(replace_acc=replace_acc)
            agg_output.update(non_replace_acc=non_replace_acc)
        
        bin_loss = sum(log.get('bin_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(bin_loss=bin_loss)
        mlm_loss = sum(log.get('mlm_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(mlm_loss=mlm_loss)

        return agg_output
