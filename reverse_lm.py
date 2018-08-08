import argparse
from generate import main as generate_main
from preprocess import get_parser as get_preprocess_parser
from preprocess import main as preprocess_main
from hierarchical_generate import main as hierarchical_generate_main
from multiprocessing_train import main as train_main
from eval_lm import main as eval_main
from shutil import copyfile

from fairseq import options 

# One function to rule them all
def main(args):
    print("STEP 0 - generate training data")
    generate_from_trained_model(args) # Should save generate datasets #TODO: make it do that!
    print("STEP 1 - preprocess this generated training data correctly")
    preprocess_training_data(args)
    print("STEP 2 - train model")
    train_model(args)
    print("STEP 3 - evaluate model")
    evaluate_model(args)
    print("STEP 4 - done!")

def generate_from_trained_model(args):
    generate_function = hierarchical_generate_main if args.hierarchical else generate_main
    # Generate data from our story generation model
    generate_parser = options.get_generation_parser()
    
    
    # One thing I wasn't certain about is what gets used as the source passed into the trained model to make our training sets.
    # Using the original training set seems weird since a model which just memorized the inputs would do well.
    # Using the val/test sets could work, only they have fewer data points.  And besides, then what do we test the model on?
    # Currently it just uses train/val/test as usual although I don't think that's optimal.
    
    generate_args = options.parse_args_and_arch(
        generate_parser,
        [
            args.data_dir,
            '--path', args.model_dir,
            '--gen-subset', 'train', #TODO: why is it asking for hierarchical-attention and stuff???
            '--save_generated_file', args.save_dir + "train"
        ]
    )
    
    generate_function(generate_args)
    generate_args = options.parse_args_and_arch(
        generate_parser,
        [
            args.data_dir,
            '--path', args.model_dir,
            '--gen-subset', 'test',
            '--save_generated_file', args.save_dir + "test"
        ]
    )    
    generate_function(generate_args)
    generate_args = options.parse_args_and_arch(
        generate_parser,
        [
            args.data_dir,
            '--path', args.model_dir,
            '--gen-subset', 'valid',
            '--save_generated_file', args.save_dir + "valid"
        ]
    )    
    generate_function(generate_args)
    
    
def preprocess_training_data(args):
    preprocess_parser = get_preprocess_parser()
    preprocess_args = options.parse_args_and_arch(
        preprocess_parser,
        [
            '-s', 'generated_source',
            '-t', 'generated_target',
            '--trainpref', args.save_dir + '/train',
            '--validpref', args.save_dir + '/valid',
            '--testpref', args.save_dir + '/test',
            '--destdir' , args.save_dir
        ],
    )
    preprocess_main(preprocess_args)
    
    
def train_model(args):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            args.save_dir, # path where dataset is saved
            '--save-dir', args.save_dir,
            '-a', 'transformer',
            '--source-lang', "generated_source",
            '--target-lang', "generated_target", #TODO: Later change this to be the generated target dict
        ],
    )
    train_main(train_args)


# This function is almost completely broken, I think.
def evaluate_model(args):
    eval_lm_parser = options.get_eval_lm_parser()
    eval_lm_args = options.parse_args_and_arch(
        eval_lm_parser,
        [
            args.data_dir, # data directory (original stories)
            '--path', args.save_dir, # model directory
            '--gen-subset', 'test',
        ]
    )   
    eval_main(eval_lm_args)



# Create and read command line options
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model on generated texts, test on real texts')
    parser.add_argument('--data-dir', default="data-bin/wikitext_source_to_target", help='directory for the test data')
    parser.add_argument('--save-dir', default="test_save_dir/",
                        help="directory where trained model and generated stories are saved (otherwise they aren't saved")
    parser.add_argument('--model-dir', default="model2_checkpoints/checkpoint_best.pt",
                        help="directory where model to generate texts is saved")
    parser.add_argument('--hierarchical', action='store_true', help="use hierarchical_generate rather than generate")
    parser.add_argument('--source_lang', default='wikitext_source', help="source lang")
    parser.add_argument('--target_lang', default='wikitext_target', help="target lang")
    args = parser.parse_args()
    main(args)