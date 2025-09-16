import random
import re
import shutil
from pathlib import Path
from template_sentences import indicative_templates, subjunctive_templates

random.seed(42)

# Expand to 1,000 examples each using mutation
def mutate(sentence: str) -> str:
    replacements = [
        (r'\bif\b',
         random.choices(["provided that", "on condition that", "assuming", "in case", "supposing"],
                        weights=[3, 1, 4, 2, 4])[0]),
        (r"(?<![a-zA-Z])she(?![a-zA-Z'])", random.choice(["they", "we", "he", "you", "the woman"])),
        (r"(?<![a-zA-Z])he(?![a-zA-Z'])", random.choice(["they", "we", "you", "she", "the man"])),
        (r"(?<![a-zA-Z])they(?![a-zA-Z'])", random.choice(["you", "we", "he", "she", "the guests"])),
        (r"(?<![a-zA-Z])we(?![a-zA-Z'])", random.choice(["they", "you", "he", "she", "all of us"])),
        (r"(?<![a-zA-Z])him(?![a-zA-Z'])", random.choice(["them", "us", "her", "you", "the woman"])),
        (r"(?<![a-zA-Z])them(?![a-zA-Z'])", random.choice(["her", "us", "him", "you", "the woman"])),
        (r"(?<![a-zA-Z])us(?![a-zA-Z'])", random.choice(["them", "her", "him", "you", "the woman"])),
        (r"(?<![a-zA-Z])I(?![a-zA-Z'])", random.choice(["they", "you"])),
    ]

    mutated = sentence
    # Randomly select 1-4 replacements to apply
    for old, new in random.sample(replacements, counts=[1, 4, 4, 4, 4, 3, 3, 3, 2], k=random.randint(1, 4)):
        # Only perform replacement if the pattern exists in the string
        if re.search(old, mutated, re.IGNORECASE):
            mutated = re.sub(old, new, mutated, count=1, flags=re.IGNORECASE)
    return mutated


# Create unique mutations of each type for more total sentences
def generate_mutations(templates, n=1000):
    # labeled = [s.strip().rstrip(".") + f". {label}" for s in templates]
    corpus = set(template for idx, template in enumerate(templates) if idx < n)
    while len(corpus) < n:
        template = random.choice(templates)
        mutated = mutate(template)
        # Ensure no duplicates (and end with period + label)
        # final = mutated.strip().rstrip(".") + f". {label}"
        corpus.add(mutated)
    return list(corpus)


# Clear and recreate directories
def clear_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # "mutate" or "stay" (!161, 1)
    mode = "mutate"

    # Clear all directories before generating new files
    clear_directory(Path("data/train/ind"))
    clear_directory(Path("data/train/subj"))
    clear_directory(Path("data/test/ind"))
    clear_directory(Path("data/test/subj"))

    if mode == "mutate":
        indicative_list = generate_mutations(indicative_templates)
        subjunctive_list = generate_mutations(subjunctive_templates)
    else:
        indicative_list = indicative_templates
        subjunctive_list = subjunctive_templates

    # Combine original and mutated, shuffle, and save
    random.shuffle(indicative_list)
    random.shuffle(subjunctive_list)

    indicative_boundary = round(0.8 * len(indicative_list))
    subjunctive_boundary = round(0.8 * len(subjunctive_list))

    # Save to file
    for idx, sen in enumerate(indicative_list[:indicative_boundary]):
        Path("data/train/ind/train_i_" + str(idx)).write_text(sen)
    for idx, sen in enumerate(subjunctive_list[:subjunctive_boundary]):
        Path("data/train/subj/train_s_" + str(idx)).write_text(sen)

    for idx, sen in enumerate(indicative_list[indicative_boundary:]):
        Path("data/test/ind/test_i_" + str(idx)).write_text(sen)
    for idx, sen in enumerate(subjunctive_list[subjunctive_boundary:]):
        Path("data/test/subj/test_s_" + str(idx)).write_text(sen)