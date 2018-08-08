import random

story_files = ['baseline_wikitext_stories.txt.generated_target', 'hierarchical_wikitext_stories.txt.generated_target', 'just_outline_wikitext_stories.txt.generated_target']
output_eval_file_prefix = "article_evaluation"
output_order_file = "story_order.txt"

story_content = []

for file in story_files:
    with open(file, "r") as f:
        story_content.extend(f.readlines())

number_of_stories = len(story_content)
number_of_stories_per_person = 10
number_of_evals = (number_of_stories - 1) // number_of_stories_per_person + 1

eval_intro = """Article Evaluation
Instructions: The articles you will be examining are intended to be like Wikipedia articles. For each article you will find three questions about the story at the end. The questions will always be the same and will assess the coherence and overall quality of the story. There are two special tokens that will appear in the stories. <num> is used to indicate a number should be present while <unk> is used to indicate a rare word (like a name) should be present. Do not judge the articles on factual accuracy as that was not intended to be achieved. Also, the articles were truncated at 1000 words so the article may end abruptly. Another article quirk is often symbols like periods/hyphens will have space surrounding them. Lastly, all articles were lower cased so don’t consider capitalization.

"""

story_questions = """Question 1 (Global Coherence) The article should be well-structured and well-organized. The article should not just be a heap of unrelated information, but should build from sentence to sentence and paragraph to paragraph. Abrupt changes in topic without any transitions are problematic.
   
1. Very Poor
2. Poor 
3. Barely Acceptable 
4. Good
5. Very Good (Human Quality)


Question 2 (Overall Quality) How realistic is the article as a whole given that the article is meant to be a Wikipedia article.

1. Very Poor
2. Poor 
3. Barely Acceptable 
4. Good
5. Very Good (Human Quality)

"""

story_content_with_indices = list(zip(range(number_of_stories), story_content))
random.shuffle(story_content_with_indices)

indices = []

for i in range(number_of_evals):
    eval_file_name = output_eval_file_prefix + str(i) + ".txt"
    with open(eval_file_name, "w") as f:
        f.write(eval_intro)

for i, (index, story) in enumerate(story_content_with_indices):
    indices.append(index)
    eval_file_name = output_eval_file_prefix + str((i // number_of_stories_per_person)) + ".txt"
    with open(eval_file_name, "a") as f:
        f.write("Story " + str(i % number_of_stories_per_person + 1) + "\n")
        f.write(story.replace("<newline>", "\n") + "\n")
        f.write(story_questions)

with open(output_order_file, "w") as f:
    f.write("\n".join(map(str, indices)))