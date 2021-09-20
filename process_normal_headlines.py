### Preproccesses collected onion and not onion headlines

import numpy as np
import json
import sys
from transformers import GPT2Tokenizer

# remember to add end of text token to end of headline
add_end_token = lambda x : x.lower() + "<|endoftext|>"
# load gpt2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

headlines = []

csv_path = "abcnews-date-text.csv"
with open(csv_path) as f:
	lines = f.readlines()
	for line in lines:
		text = line.split(",")[1]
		headlines += [tokenizer.encode(add_end_token(text))]

# create set
headlines = np.array(headlines)
headlines = np.unique(headlines)

# save dataset
np.save("processed_headlines/normal.npy", headlines)
