#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse
from itertools import zip_longest
import os
import shutil

from fairseq.data import indexed_dataset, dictionary
from fairseq.tokenizer import build_tokenizer

def get_parser():
    parser = argparse.ArgumentParser(
        description='Data pre-processing: Create dictionary and store data in binary format')
    parser.add_argument('-s', '--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET', help='target language')
    parser.add_argument('--trainpref', metavar='FP', default=None, help='target language')
    parser.add_argument('--validpref', metavar='FP', default=None, help='comma separated, valid language prefixes')
    parser.add_argument('--testpref', metavar='FP', default=None, help='comma separated, test language prefixes')
    parser.add_argument('--destdir', metavar='DIR', default='data-bin', help='destination dir')
    parser.add_argument('--thresholdtgt', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--thresholdsrc', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--tgtdict', metavar='FP', help='reuse given target dictionary')
    parser.add_argument('--srcdict', metavar='FP', help='reuse given source dictionary')
    parser.add_argument('--nwordstgt', metavar='N', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--nwordssrc', metavar='N', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--alignfile', metavar='ALIGN', default=None, help='an alignment file (optional)')
    parser.add_argument('--output-format', metavar='FORMAT', default='binary', choices=['binary', 'raw'],
                        help='output format (optional)')
    parser.add_argument('--joined-dictionary', action='store_true', help='Generate joined dictionary')
    parser.add_argument('--only-source', action='store_true', help='Only process the source language')
    parser.add_argument('--padding-factor', metavar='N', default=8, type=int,
                        help='Pad dictionary size to be multiple of N')
    parser.add_argument('--tokenizer_name', metavar='N', default='default', choices=['default', 'nltk', 'sacremoses'],
                        help="Which tokenizer to use. Choices are default, nltk, sacremoses. default tokenizes by splitting on white space. nltk uses "
                             "nltk's word_tokenize which better takes into account punctuation. As an example "
                             "'Hello, how's your day today?' would be tokenized as "
                             "['Hello,' , 'how's', 'your', 'day', 'today?'] when using the default, but would instead be tokenized as "
                             "['Hello', ',', 'how', ''s', 'your', 'day', 'today', '?'] when using nltk. The sacremoses tokenizer is from this package, "
                             "https://github.com/alvations/sacremoses.")
    parser.add_argument('--max-source-length', metavar='N', type=int,
                        help="The maximum length of the source. If a sequence is longer it will be truncated to this length. " 
                             "If -1, then no truncation occurs. For joined_dictionary's the maximum of the source/target max length will "
                             " be used to build the dictionary. Default is no truncation.")
    parser.add_argument('--max-target-length', metavar='N', type=int,
                        help="The maximum length of the target. If a sequence is longer it will be truncated to this length. " 
                             "If -1, then no truncation occurs. For joined_dictionary's the maximum of the source/target max length will "
                             "be used to build the dictionary. Default is no truncation.")
    return parser


def main(args):
    print(args)
    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source
    source_tokenizer = build_tokenizer(args, args.max_source_length)
    target_tokenizer = build_tokenizer(args, args.max_target_length)

    def build_dictionary(tokenizer, filenames, max_length=None):
        d = dictionary.Dictionary()
        for filename in filenames:
            tokenizer.add_file_to_dictionary(filename, d)
        return d

    def train_path(lang):
        return '{}{}'.format(args.trainpref, ('.' + lang) if lang else '')

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += f'.{lang}'
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path('dict', lang) + '.txt'

    def dataset_dest_path(output_prefix, lang, extension):
        base = f'{args.destdir}/{output_prefix}'
        lang_part = f'.{args.source_lang}-{args.target_lang}.{lang}' if lang is not None else ''
        return f'{base}{lang_part}.{extension}'

    if args.joined_dictionary:
        assert not args.srcdict, 'cannot combine --srcdict and --joined-dictionary'
        assert not args.tgtdict, 'cannot combine --tgtdict and --joined-dictionary'

        if args.max_source_length is None or args.max_target_length is None:
            tokenizer = source_tokenizer
        elif args.max_source_length > args.max_target_length:
            tokenizer = source_tokenizer
        else:
            tokenizer = target_tokenizer

        src_dict = build_dictionary(tokenizer, set([
            train_path(lang)
            for lang in [args.source_lang, args.target_lang]
        ]))
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = dictionary.Dictionary.load(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(source_tokenizer, [train_path(args.source_lang)])
        if target:
            if args.tgtdict:
                tgt_dict = dictionary.Dictionary.load(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary(target_tokenizer, [train_path(args.target_lang)])

    src_dict.finalize(
        threshold=args.thresholdsrc,
        nwords=args.nwordssrc,
        padding_factor=args.padding_factor,
    )
    src_dict.save(dict_path(args.source_lang))
    if target:
        if not args.joined_dictionary:
            tgt_dict.finalize(
                threshold=args.thresholdtgt,
                nwords=args.nwordstgt,
                padding_factor=args.padding_factor,
            )
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(tokenizer, input_prefix, output_prefix, lang):
        dict = dictionary.Dictionary.load(dict_path(lang))
        print('| [{}] Dictionary: {} types'.format(lang, len(dict) - 1))

        ds = indexed_dataset.IndexedDatasetBuilder(dataset_dest_path(output_prefix, lang, 'bin'))

        def consumer(tensor):
            ds.add_item(tensor)

        input_file = '{}{}'.format(input_prefix, ('.' + lang) if lang is not None else '')
        res = tokenizer.binarize(input_file, dict, consumer)
        print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, res['nseq'], res['ntok'],
            100 * res['nunk'] / res['ntok'], dict.unk_word))
        ds.finalize(dataset_dest_path(output_prefix, lang, 'idx'))

    def make_dataset(tokenizer, input_prefix, output_prefix, lang):
        if args.output_format == 'binary':
            make_binary_dataset(tokenizer, input_prefix, output_prefix, lang)
        elif args.output_format == 'raw':
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + '.{}-{}'.format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_all(tokenizer, lang):
        if args.trainpref:
            make_dataset(tokenizer, args.trainpref, 'train', lang)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(',')):
                outprefix = 'valid{}'.format(k) if k > 0 else 'valid'
                make_dataset(tokenizer, validpref, outprefix, lang)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(',')):
                outprefix = 'test{}'.format(k) if k > 0 else 'test'
                make_dataset(tokenizer, testpref, outprefix, lang)

    make_all(source_tokenizer, args.source_lang)
    if target:
        make_all(target_tokenizer, args.target_lang)

    print('| Wrote preprocessed data to {}'.format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        src_dict = dictionary.Dictionary.load(dict_path(args.source_lang))
        tgt_dict = dictionary.Dictionary.load(dict_path(args.target_lang))
        freq_map = {}
        with open(args.alignfile, 'r') as align_file:
            with open(src_file_name, 'r') as src_file:
                with open(tgt_file_name, 'r') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = source_tokenizer.tokenize(s, src_dict, add_if_not_exist=False)
                        ti = target_tokenizer.tokenize(t, tgt_dict, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split('-')), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(os.path.join(args.destdir, 'alignment.{}-{}.txt'.format(
                args.source_lang, args.target_lang)), 'w') as f:
            for k, v in align_dict.items():
                print('{} {}'.format(src_dict[k], tgt_dict[v]), file=f)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
