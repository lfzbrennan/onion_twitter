### Preproccesses collected onion and not onion headlines

import numpy as np
import json
import sys
from transformers import GPT2Tokenizer

# remember to add end of text token to end of headline
add_end_token = lambda x : x.lower() + "<|endoftext|>"

# paths
not_onion_path = "raw_headlines/not_onion.npy"
onion_path = "raw_headlines/onion.npy"

not_onion = np.load(not_onion_path)
onion = np.load(onion_path)


# load gpt2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# combined tokenized headlines into 1 array after adding end of text token
combined = []
for i in range(len(not_onion)):
    combined += [tokenizer.encode(add_end_token(not_onion[i]))]

for i in range(len(onion)):
    combined += [tokenizer.encode(add_end_token(onion[i]))]

print(len(combined))

# add data from csvfile extracted from https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/data?select=Sarcasm_Headlines_Dataset_v2.json
with open("Sarcasm_Headlines_Dataset_v2.json") as f:
    lines = f.readlines()
    for line in lines:
        cur_data = json.loads(line)
        if cur_data['is_sarcastic'] == 1:
            combined += [tokenizer.encode(add_end_token(cur_data['headline']))]

print(len(combined))
# create set
combined = np.array(combined)
combined = np.unique(combined)

# save dataset
np.save("processed_headlines/combined.npy", combined)
