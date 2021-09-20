import numpy as np 

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("output/checkpoint-35000")

add_end_token = lambda x : x.lower() + "<|endoftext|>"

real = np.load("processed_headlines/combined.npy", allow_pickle=True)
fake = np.load("generated_headlines.npy", allow_pickle=True)

out = []

for r in real:
	out += [(r, 1)]

for f in fake:
	out += [(tokenizer.encode(add_end_token(f)), 0)]

out = np.array(out, dtype=object)
np.random.shuffle(out)

np.save("processed_headlines/critic_dataset.npy", out)