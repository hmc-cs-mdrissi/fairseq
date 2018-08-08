# Imports
import nltk
import argparse
import tqdm
import multiprocessing

NEWLINE_TOKEN = "<newline>"

def postprocess_sentence_copy_stories(inputs):
    outline, story = inputs
    outline_sentences = outline.split("<newline>")

    def put_matching_sentence(matchobj):
        return outline_sentences[matchobj.group(1)]

    return re.sub("<sentence_(\d+)>", put_matching_sentence, story)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Generation')
    parser.add_argument('--input-story-file-name', help='Name of input story file.')
    parser.add_argument('--input-outline-file-name', help='Name of input outline file.')
    parser.add_argument('--output-file-name', help='Name of output file.')
    parser.add_argument('--chunksize', type=int, default=100, help="Size of chunks to use for each process. Default is 100.")
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    args = parser.parse_args()
    print("Currently postprocessing sentence copy stories for " + args.input_story_file_name)

    with open(args.input_story_file_name) as f:
        raw_stories = f.readlines()

    with open(args.input_outline_file_name) as f:
        raw_outlines = f.readlines()

    inputs = zip(raw_outlines, raw_stories) 

    with open(args.output_file_name, 'w') as f:
        for story in tqdm.tqdm(p.imap(postprocess_sentence_copy_stories, inputs, chunksize=args.chunksize)):
            f.write(story + "\n")
