import sys
import json
import time
import pickle
import argparse
import urllib.request


def add_dp(question, answer, dataset):
    dataset.append(
        {"question": question,
         "gold-answer": " " + answer
         })


parser = argparse.ArgumentParser(description='Process Arguments for experiments with GPTJ LLM on CounterFact')
parser.add_argument('--path', type=str, default="./counterfact", help='place where to save the dataset')
args = parser.parse_args()

time_s = time.time()
rome_path = "https://rome.baulab.info/data/dsets/counterfact.json"
print(f"Fetching Counterfact data from {rome_path}")

if sys.version_info[0] == 2:
    response = urllib.urlopen(rome_path)
    orig_dataset = json.loads(response.read())
elif sys.version_info[0] == 3:
    with urllib.request.urlopen(rome_path) as url:
        orig_dataset = json.load(url)
else:
    raise AssertionError("Unhandled python version number")
print(f"Dataset fetched in {time.time() - time_s:.3f} seconds.")

print(f"The original dataset has {len(orig_dataset)} many datapoints.")
dataset = []
for dp in orig_dataset:
    question = dp["requested_rewrite"]["prompt"].format(dp["requested_rewrite"]["subject"])
    paraphrases = dp["paraphrase_prompts"]
    assert len(paraphrases) == 2, f"Expected 2 paraphrases per questions but instead found {len(paraphrases)}."
    answer = dp["requested_rewrite"]["target_true"]["str"]

    add_dp(question, answer, dataset)
    for paraphrase in paraphrases:
        add_dp(paraphrase, answer, dataset)

print(f"After processing, the new dataset has {len(dataset)} many datapoints.")

with open(args.path, "wb") as f:
    pickle.dump(dataset, f)
