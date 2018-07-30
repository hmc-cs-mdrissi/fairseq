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
        indices = indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.paragraph_counts[indices], kind='mergesort')]

    def newline(self):
        return self.newline_idx

    def get_paragraph_count(self, idx):
        return self.paragraph_counts[idx]

def hierarchical_story_collate(samples, pad_idx, eos_idx, left_pad_source=True, 
    left_pad_outline=False, left_pad_target=False):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    outline = merge('outline', left_pad_outline=left_pad_outline)
    # we create a shifted version of outline for feeding the
    # previous output token(s) into the next first decoder step
    prev_outline_output_tokens = merge(
        'outline',
        left_pad=left_pad_outline,
        move_eos_to_beginning=True,
    )
    prev_outline_output_tokens = prev_outline_output_tokens.index_select(0, sort_order)
    outline = outline.index_select(0, sort_order)

    target = merge('target', left_pad=left_pad_target)
    # we create a shifted version of targets for feeding the
    # previous output token(s) into the next second decoder step
    prev_target_output_tokens = merge(
        'target',
        left_pad=left_pad_target,
        move_eos_to_beginning=True,
    )
    prev_target_output_tokens = prev_target_output_tokens.index_select(0, sort_order)
    target = target.index_select(0, sort_order)
    ntokens = sum(len(s['target']) for s in samples)

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_outline_output_tokens': prev_outline_output_tokens,
            'prev_target_output_tokens': prev_target_output_tokens,
        },
        'target':{
            'outline': outline,
            'target': target,
        }, 
    }    

class HierarchicalStoryDataset(StoryDataset):
    """A triple of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        outline, outline_sizes, outline_dict,
        tgt, tgt_sizes, tgt_dict,
        left_pad_source=True, left_pad_outline=False, left_pad_target=False,
        max_source_positions=1024, max_outline_positions=1024, max_target_positions=1024,
        shuffle=True,
    ):
        assert src_dict.pad() == outline_dict.pad() and src_dict.pad() == tgt_dict.pad() 
        assert src_dict.eos() == outline_dict.eos() and src_dict.eos() == tgt_dict.eos() 
        assert src_dict.unk() == outline_dict.unk() and src_dict.unk() == tgt_dict.unk() 
        assert src_dict.index("<newline>") != src_dict.unk()
        self.src = src
        self.outline = outline
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.outline_sizes = np.array(outline_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.src_dict = src_dict
        self.outline_dict = outline_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_outline = left_pad_outline
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_outline_positions = max_outline_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.newline_idx = src_dict.index("<newline>")
        self.paragraph_counts = np.array(list(map(lambda datapoint: (datapoint == self.newline_idx).sum().item() + 1, outline)))

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'outline': self.outline[index],
            'target': self.tgt[index]
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return hierarchical_story_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_outline=left_pad_outline, 
            left_pad_target=self.left_pad_target,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, outline_len=128, tgt_len=128):
        max_source_positions, max_outline_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, outline_len, tgt_len = min(src_len, max_source_positions), min(outline_len, max_outline_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, outline_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'outline': self.outline_dict.dummy_sentence(outline_len)
                'target': self.tgt_dict.dummy_sentence(tgt_len)
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.outline_sizes[index], self.tgt_sizes[index])

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        indices = indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.outline_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.paragraph_counts[indices], kind='mergesort')]

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_outline_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and self.outline_sizes[index] <= max_outline_positions
            and self.tgt_sizes[index] <= max_target_positions
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_outline_positions, self.max_target_positions
        assert len(max_positions) == 3
        max_src_pos, max_outline_positions, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_outline_positions, max_outline_positions), 
               min(self.max_target_positions, max_tgt_pos)