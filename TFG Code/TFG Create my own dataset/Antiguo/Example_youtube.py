from datasets import load_dataset
import json
reddit = load_dataset('jamescalam/reddit-python', split='train')

count = 0
for record in reddit:
    print(record)
    count += 1
    if count == 5:
        break
# %% Transform dataset in a DataFrame (Theres is json, pandas...)
reddit = reddit.to_pandas()

# %%
reddit = reddit.to_dict(orient="records")

# %%
with open("train.jsonl", "w") as f:
    for line in reddit:
        f.write(json.dumps(line)+"\n")
