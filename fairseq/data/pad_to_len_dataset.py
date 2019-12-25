# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class PadToLenDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, left_pad, pad_len):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_len = pad_len

    def collater(self, samples):
        return collate_tokens_len(samples, self.pad_idx, self.pad_len, left_pad=self.left_pad)


def collate_tokens_len(values, pad_idx, pad_len, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    assert size <= pad_len
    size = pad_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res