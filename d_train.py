import transformers
import torch
from tqdm import tqdm, trange
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os


from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from transformers import GPT2Config, GPT2Model, AdamW, get_linear_schedule_with_warmup, GPT2PreTrainedModel, GPT2Tokenizer

class D_Dataset(Dataset):
	def __init__(self, train="train"):
		self.examples = np.load("processed_headlines/critic_dataset.npy", allow_pickle=True)
	def __getitem__(self, idx):
		tokens, label = self.examples[idx]
		return torch.LongTensor(tokens), label
	def __len__(self):
		return len(self.examples)

class D_Model(GPT2PreTrainedModel):
	def __init__(self, config):
		# create base gpt2 model
		super().__init__(config)
		self.gptmodel = GPT2Model(config)

		# add one hidden layer
		self.vuln_out1 = nn.Linear(config.n_embd, config.n_embd)
		# output layer
		self.vuln_out3 = nn.Linear(config.n_embd, 1)

	def forward(self, inputs):
		# get last hidden state of transformer output
		gptout = (self.gptmodel(inputs)[0])[:, -1, :].squeeze(-1)
		# push through linear model
		out1 = F.relu(self.vuln_out1(gptout))
		out3 = self.vuln_out3(out1)

		return out3

	# load base gpt2 model
	def load_gpt(self, gpt2_path):
		cur_state_dict = self.gptmodel.state_dict()
		gpt_state_dict = torch.load(gpt2_path)
		for k in cur_state_dict:
			cur_state_dict[k] = gpt_state_dict[f"transformer.{k}"]
		self.gptmodel.load_state_dict(cur_state_dict)


def save_model(vulnhead_model, save_dir):
	print(f"Saving models into {save_dir}")
	torch.save(vulnhead_model.state_dict(), f"{save_dir}/d_model.pt")


def eval():

	device = torch.device("cuda")

	print("Building config...")
	model_config = GPT2Config(
		vocab_size= 50257,
		n_positions=1024,
		n_ctx=1024,
		n_embd=1024,
		n_layer=24,
		n_head=16,
		n_inner=None,
		activation_function="gelu_new",
		resid_pdrop=0.1,
		embd_pdrop=0.1,
		attn_pdrop=0.1,
		layer_norm_epsilon=1e-5,
		initializer_range=0.02,
		summary_type="cls_index",
		summary_use_proj=True,
		summary_activation=None,
		summary_proj_to_labels=True,
		summary_first_dropout=0.1,
		bos_token_id=50256,
		eos_token_id=50256,
		gradient_checkpointing=False,

		)
	print("Building model...")
	model = D_Model(model_config)

	gpt_model_path = "d_output/checkpoint-940000/d_model.pt"
	model.load_state_dict(torch.load(gpt_model_path))
	model.eval()
	model.to(device)

	print("Loading tokenizer...")

	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

	out_headlines = []
	print("Loading headlines...")
	all_generated = np.load("generated_headlines2.npy")
	print("Starting eval...")
	for headline in all_generated:
		inputs = torch.LongTensor(tokenizer.encode(headline)).unsqueeze(0)
		inputs = inputs.to(device)
		with torch.no_grad():
			outputs = model(inputs).squeeze()
			outputs = outputs - 17.0
			if outputs > 0:
				out_headlines += [headline]


	np.save("good_headlines.npy", np.array(out_headlines))

def train():
	output_dir = "d_output"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	save_steps = 5000
	logging_steps = 5000

	train_epochs = 4
	max_grad_norm = 1.0

	global_step = 0
	epochs_trained = 0
	steps_trained_in_current_epoch = 0
	batch_size = 1

	tr_loss, logging_loss = 0.0, 0.0

	learning_rate = 1e-4
	adam_epsilon = 1e-8

	device = torch.device("cuda")

	print("Building config...")
	model_config = GPT2Config(
		vocab_size= 50257,
		n_positions=1024,
		n_ctx=1024,
		n_embd=1024,
		n_layer=24,
		n_head=16,
		n_inner=None,
		activation_function="gelu_new",
		resid_pdrop=0.1,
		embd_pdrop=0.1,
		attn_pdrop=0.1,
		layer_norm_epsilon=1e-5,
		initializer_range=0.02,
		summary_type="cls_index",
		summary_use_proj=True,
		summary_activation=None,
		summary_proj_to_labels=True,
		summary_first_dropout=0.1,
		bos_token_id=50256,
		eos_token_id=50256,
		gradient_checkpointing=False,

		)
	print("Building model...")
	model = D_Model(model_config)

	gpt_model_path = "output/checkpoint-35000/pytorch_model.bin"

	# load pretrained c++ transformers for tranfer learning
	model.load_gpt(gpt_model_path)
	model.to(device)

	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

	print("Creating DataLoaders...")
	train_dataset = D_Dataset("train")

	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, num_workers=1, sampler=train_sampler, batch_size=batch_size)

	train_iterator = trange(epochs_trained, int(train_epochs), desc="Epoch")

	print("Creating optimizer...")
	optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) // train_epochs)

	# either vulnerable or not vulnerable, so BCE loss
	criterion = nn.BCEWithLogitsLoss()

	print("Starting training...")
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration")
		# train
		for step, batch in enumerate(epoch_iterator):

			# since using gpt (not bert or elmo), we dont use masking
			inputs, labels = batch
			inputs = inputs.to(device)
			labels = labels.to(device).squeeze()
			model.train()
			outputs = model(inputs).squeeze()
			loss = criterion(outputs, labels.float()) # model outputs are always tuple in transformers (see doc)
			loss.backward()

			tr_loss += loss.item()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optimizer.step()
			scheduler.step()  # Update learning rate schedule
			model.zero_grad()
			global_step += 1
			if save_steps > 0 and global_step % save_steps == 0:
				checkpoint_prefix = 'checkpoint'
					   # Save model checkpoint
				save_dir = os.path.join(output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)
				save_model(model, save_dir)
				
	# final output
	save_dir = "final"
	save_dir = os.path.join(output_dir, save_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_model(model.module, save_dir)

if __name__ == "__main__":
	eval()
