# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from fairseq import utils
from . import (
    BaseFairseqModel, register_model,
)

@register_model('hierarchical_sequence_model')
class HierarchicalSequenceModel(BaseFairseqModel):
    def __init__(self, first_model, second_model):
        super().__init__()
        self.first_model = first_model
        self.second_model = second_model
        assert(issubclass(first_model, BaseFairseqModel))
        assert(issubclass(second_model, BaseFairseqModel))

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--first_model', metavar='DIR',
                            help='path to load checkpoint from first model.')
        parser.add_argument('--second_model', metavar='DIR',
                            help='path to load checkpoint from second_model.')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # not actually for inference, but loads model parameters
        first_model, second_model = utils.load_ensemble_for_inference(filenames=[args.first_model, args.second_model], task=task)[0]
        model = HierarchicalSequenceModel(first_model, second_model)
        return model

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        raise ValueError("1. Hi. Curious where this is used.")
        return sample['net_target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        raise ValueError("2. Hi. Curious where this is used.")
        return self.second_model.decoder.get_normalized_probs(net_output, log_probs, sample)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.first_model.encoder.max_positions(), 
               min(self.first_model.decoder.max_positions(), self.second_model.encoder.max_positions()),
               self.second_model.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        raise ValueError("3. Hi. Curious where this is used.")
        return self.second_model.decoder.max_positions()