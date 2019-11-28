# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
import math

from fairseq.data import data_utils, Dictionary

from . import BaseWrapperDataset, LRUCacheDataset


class EditTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        (to-do: edit) return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_edit(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_type='source')),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_type='target')),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_type='operation')),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        replace_idx: int,
        return_type: str = "source",
        seed: int = 1,
        mask_prob: float = 0.15,
        delete_prob: float = 0.02,
        swap_prob: float = 0.01,
        random_replace_prob: float = 0.15,
        random_replace: bool = False,
        # leave_unmasked_prob: float = 0.1,
        # random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        # mask_whole_words: torch.Tensor = None,
    ):
        assert 0.0 <= mask_prob < 1.0
        # assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= random_replace_prob <= 1.0
        # assert 0.0 <= leave_unmasked_prob <= 1.0
        # assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.replace_idx = replace_idx
        self.seed = seed
        self.mask_prob = mask_prob
        self.delete_prob = delete_prob
        self.swap_prob = swap_prob
        self.random_replace_prob = random_replace_prob
        self.random_replace = random_replace
        self.return_type = return_type
        # self.leave_unmasked_prob = leave_unmasked_prob
        # self.random_token_prob = random_token_prob
        # self.mask_whole_words = mask_whole_words

        if freq_weighted_replacement:
            weights = np.array(self.vocab.count)
        else:
            weights = np.ones(len(self.vocab))
        weights[:self.vocab.nspecial] = 0
        # do not replace to mask_idx
        weights[self.mask_idx] = 0
        self.weights = weights / weights.sum()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            assert self.mask_idx not in item, \
                'Dataset contains mask_idx (={}), this is not expected!'.format(
                    self.mask_idx,
                )

            delete_prob = (1 - math.sqrt(1-4*self.delete_prob))/2 #self.delete_prob + self.delete_prob * self.delete_prob
            delete_num = int(
                # add a random number for probabilistic rounding
                delete_prob * len(item) + np.random.rand()
            )

            new_item = np.copy(item)
            operation = np.zeros_like(new_item, dtype=np.int8)
            # mask_prob, replace_prob, add_prob, delete_prob, make add_prob = delte_prob;
            if delete_num > 0 and len(new_item) > 2:
                # delete:2
                delete_pos = np.sort(np.random.choice(len(new_item)-2, delete_num, replace=False))+1 # cannot delete the 1st and last token
                item_segments = [new_item[:delete_pos[0]]]
                op_segments = [operation[:delete_pos[0]]]
                for idx,pos in enumerate(delete_pos):
                    if idx != len(delete_pos)-1:
                        ed = delete_pos[idx+1]
                    else:
                        ed = len(new_item)
                    if pos+1 == ed: # skip the neighbor positions
                        delete_num -= 1
                        item_segments.append(new_item[pos:ed])
                        op_segments.append(operation[pos:ed])
                        continue
                    op_segments[-1][-1] = 2
                    item_segments.append(new_item[pos+1:ed])
                    op_segments.append(operation[pos+1:ed])
                new_item = np.concatenate(item_segments, axis=0)
                operation = np.concatenate(op_segments, axis=0)
                # print("step-1",operation)
            # add:3
            if delete_num > 0 and len(new_item) > 2:
                add_pos = np.sort(np.random.choice(len(new_item)-2, delete_num, replace=False))+1 # cannot add after the last token
                item_segments = [new_item[:add_pos[0]]]
                op_segments = [operation[:add_pos[0]]]
                add_rand_tokens = np.random.choice(len(self.vocab), len(add_pos), p=self.weights)
                for idx,pos in enumerate(add_pos):
                    item_segments.append(add_rand_tokens[idx:idx+1])
                    if operation[pos-1] == 0:
                        op_segments.append(np.array([3])) # set operation
                    else:
                        op_segments[-1][-1] = 0
                        op_segments.append(np.array([1])) # detele and then add -> replace:3
                    if idx != len(add_pos)-1:
                        ed = add_pos[idx+1]
                    else:
                        ed = len(new_item)
                    item_segments.append(new_item[pos:ed])
                    op_segments.append(operation[pos:ed])
                new_item = np.concatenate(item_segments, axis=0)
                operation = np.concatenate(op_segments, axis=0)
            
            if self.return_type == "target":
                return torch.from_numpy(new_item)
            
            sz  = len(operation)
            # decide elements to mask
            mask = np.full(sz, False)
            mask_prob = self.mask_prob / (1 - 2 * delete_prob + delete_prob * delete_prob)
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * len(item) + np.random.rand()
            )
            if self.random_replace_prob > 0.0:
                replace_prob = self.random_replace_prob / (1 - 2 * delete_prob + delete_prob * delete_prob) - delete_prob * delete_prob
                num_replace = int(
                    # add a random number for probabilistic rounding
                    replace_prob * len(item) + np.random.rand()
                )
            else:
                num_replace = 0
            
            if self.swap_prob > 0.0:
                swap_prob = self.swap_prob / (1 - 2 * delete_prob + delete_prob * delete_prob) - delete_prob * delete_prob
                num_swap = int(
                    # add a random number for probabilistic rounding
                    swap_prob * len(item) + np.random.rand()
                ) * 2
            else:
                num_swap = 0

            mask_replace_pos = np.random.choice(sz, num_mask + num_replace + num_swap, replace=False)
            mask[mask_replace_pos[:num_mask]] = True
            mask = (mask & (operation==0))

            if self.random_replace_prob > 0.0:
                replace = np.full(sz, False)
                replace[mask_replace_pos[num_mask : num_mask + num_replace]] = True
                replace = (replace & (operation==0)) | (operation == 1)
            else:
                replace = None

            # mask:5
            new_item[mask] = self.mask_idx
            operation[mask] = 5
            if replace is not None:
                num_replace = replace.sum()
                if num_replace > 0:
                    if self.random_replace:
                        new_item[replace] = np.random.choice(
                            len(self.vocab),
                            num_replace,
                            p=self.weights,
                        )
                    else:
                        new_item[replace] = self.replace_idx
                    operation[replace] = 1 # replace:1
            # if sum(new_item==self.replace_idx)!=sum(operation==4):
            # print(self.return_type, sum(new_item==self.replace_idx), sum(operation==4))

            if self.swap_prob > 0.0:
                # swap:4
                swap_pos = mask_replace_pos[num_mask + num_replace:]
                idx = 0
                while idx+1 < len(swap_pos):
                    temp = new_item[swap_pos[idx]]
                    new_item[swap_pos[idx]] = new_item[swap_pos[idx+1]]
                    new_item[swap_pos[idx+1]] = temp
                    idx += 2
                operation[swap_pos[:idx+2]] = 4

            if self.return_type == "source":
                return torch.from_numpy(new_item)
            else:
                return torch.from_numpy(operation).byte()

