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
        replace_idx = self.task.dictionary.index('<replace>')
        unmask_tokens = (sample['operation']!=5) # to-do: optimize the operation define
        not_pad_tokens = sample['net_input']['src_tokens'].ne(self.padding_idx)

        if self.args.self_replace:
            with torch.no_grad():
                replaced_tokens = sample['net_input']['src_tokens'].eq(replace_idx)
                if replaced_tokens.int().sum() > 0:
                    gen_samples = sample['target'].clone()
                    gen_samples[replaced_tokens] = mask_idx

                    mlm_logits, _ = model(gen_samples, mlm_tokens=replaced_tokens, bin_tokens=None)[0]
                    sample_probs = torch.softmax(mlm_logits, -1, dtype=torch.float32).view(-1, mlm_logits.size(-1)).detach()
                    sample_res = torch.multinomial(sample_probs, 1).view(-1)
                    sample['net_input']['src_tokens'][replaced_tokens] = sample_res

        elif not self.args.random_replace:
            replaced_tokens = sample['net_input']['src_tokens'].eq(replace_idx)
            masked_tokens = sample['net_input']['src_tokens'].eq(mask_idx)
            original_tokens = sample['target'][replaced_tokens].view(-1)
            sample['net_input']['src_tokens'] = model.replace_preprocess(sample['net_input']['src_tokens'], original_tokens,
                                                                        len(self.task.dictionary), mask_idx, 
                                                                        self.padding_idx, replace_idx, 
                                                                        replaced_tokens.view(-1).nonzero().view(-1), 
                                                                        masked_tokens.view(-1).nonzero().view(-1),
                                                                        (~not_pad_tokens).view(-1).nonzero().view(-1)).detach()

        # update replace operation
        fake_replaced_tokens = sample['net_input']['src_tokens'].eq(sample['target']) & (sample['operation'] == 1)
        sample['operation'][fake_replaced_tokens] = 0

        mlm_tokens = (sample['operation']==5)
        if self.args.predict_replace:
            mlm_tokens = mlm_tokens | (sample['operation']==1)

        mlm_sample_size = (mlm_tokens & not_pad_tokens).int().sum().item() 

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if mlm_sample_size == 0:
            mlm_tokens = None
            unmask_tokens = not_pad_tokens
        else:
            mlm_tokens = mlm_tokens & not_pad_tokens
            unmask_tokens = unmask_tokens & not_pad_tokens

        unmask_sample_size = unmask_tokens.int().sum().item()

        mlm_logits, unmask_logits = model(**sample['net_input'], mlm_tokens=mlm_tokens, bin_tokens=unmask_tokens)[0]
        targets = model.get_targets(sample, [mlm_logits])
        unmask_targets = sample['operation'][unmask_tokens]
        
        if self.args.random_replace_prob > 0.0:
            unmask_loss = F.nll_loss(
                F.log_softmax(
                    unmask_logits.view(-1, unmask_logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                unmask_targets.view(-1).long(),
                reduction='sum'
            )
        else:
            unmask_loss = torch.tensor(0.0)
        
        if mlm_sample_size != 0:
            mlm_targets = targets[mlm_tokens]

            mlm_loss = F.nll_loss(
                F.log_softmax(
                    mlm_logits.view(-1, mlm_logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                mlm_targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            loss = mlm_loss + self.args.loss_lamda * unmask_loss * mlm_sample_size / unmask_sample_size
        else:
            mlm_loss = torch.tensor(0.0)
            loss = self.args.loss_lamda * unmask_loss / unmask_sample_size

        mlm_sample_size = mlm_sample_size if mlm_sample_size!=0 else 1
        corrupted_sample_size = (sample['operation'] != 0).int().sum().item()
        logging_output = {
            'loss': utils.item(unmask_loss.data) if reduce else unmask_loss.data,
            'nll_loss': utils.item(mlm_loss.data) if reduce else mlm_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'mlm_sample_size': mlm_sample_size,
            'unmask_sample_size': unmask_sample_size,
            'corrupted_sample_size': corrupted_sample_size,
        }
        if self.args.random_replace_prob > 0.0:
            unmask_preds = unmask_logits.max(dim=1)[1]
            logging_output.update(
                unmask_ncorrect=(unmask_preds == unmask_targets).sum().item()
            )
        if mlm_sample_size > 1:
            mlm_preds = mlm_logits.max(dim=1)[1]
            logging_output.update(
                mlm_ncorrect=(mlm_preds == mlm_targets).sum().item()
            )

        return loss, mlm_sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        mlm_sample_size = sum(log.get('mlm_sample_size', 0) for log in logging_outputs)
        unmask_sample_size = sum(log.get('unmask_sample_size', 0) for log in logging_outputs)
        corrupted_sample_size = sum(log.get('corrupted_sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / unmask_sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / mlm_sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'mlm_sample_size': mlm_sample_size,
            'unmask_sample_size': unmask_sample_size,
            'corrupted_sample_size': corrupted_sample_size,
            'corrupted_rate': corrupted_sample_size/ntokens,
            'mask_replace_rate': mlm_sample_size/ntokens,
        }

        if 'unmask_ncorrect' in logging_outputs[0]: 
            unmask_ncorrect_sum = sum(log.get('unmask_ncorrect', 0) for log in logging_outputs)
            agg_output.update(ops_acc=unmask_ncorrect_sum/unmask_sample_size)
        if 'mlm_ncorrect' in logging_outputs[0]: 
            mlm_ncorrect_sum = sum(log.get('mlm_ncorrect', 0) for log in logging_outputs)
            agg_output.update(mlm_acc=mlm_ncorrect_sum/mlm_sample_size)

        return agg_output
