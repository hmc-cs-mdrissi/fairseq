# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np

from . import LanguagePairDataset, FairseqDataset

class StoryDataset(FairseqDataset):
    def newline(self):
        """Returns the index of the newline symbol."""
        raise NotImplementedError

    def get_paragraph_count(self, idx):
        """Returns the number of paragraphs for the element at location, idx."""
        raise NotImplementedError


class StoryOutlineDataset(LanguagePairDataset, StoryDataset):
    """A pair of torch.utils.data.Datasets that also orders elements by number of paragraphs in outline. Primarily necessary,
       as some of the story related models have a constraint that the number of paragraphs must be fixed for any batch. Assumes
       that the newline token is <newline>.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
    ):
        super().__init__(src, src_sizes, src_dict, tgt=tgt, tgt_sizes=tgt_sizes,
                         tgt_dict=tgt_dict, left_pad_source=left_pad_source, 
                         left_pad_target=left_pad_target, max_source_positions=max_source_positions,
                         max_target_positions=max_target_positions, shuffle=shuffle)
        self.newline_idx = src_dict.index("<newline>")
        self.paragraph_counts = np.array(list(map(lambda datapoint: (datapoint == self.newline_idx).sum().item() + 1, src)))

    # Dummy batches will always have precisely two paragraphs. While things could be tweaked so that each dummy
    # batch has a random number of paragraphs (consistent throughout the batch), I don't currently see a need for it.
    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, tgt_len)

        def get_dummy_sentence(dictionary, length):
            result_sentence = dictionary.dummy_sentence(length)
            result_sentence[result_sentence == self.newline_idx] = dictionary.unk()
            result_sentence[-2] = self.newline_idx
            return result_sentence

        return self.collater([
            {
                'id': i,
                'source': get_dummy_sentence(self.src_dict, src_len),
                'target': get_dummy_sentence(self.tgt_dict, tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.paragraph_counts[indices], kind='mergesort')]

    def newline(self):
        return self.newline_idx

    def get_paragraph_count(self, idx):
        return self.paragraph_counts[idx]

class HierarchicalStoryDataset(FairseqDataset):
    pass