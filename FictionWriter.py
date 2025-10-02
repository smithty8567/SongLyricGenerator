import random
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import CorpusGenerator as cg

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# Use max_split_size_mb:512 possibly instead or max_split_size_mb:256


class SongData(Dataset):

    def __init__(self, max_length=4, random_sample=True):
        self.corpus1, self.corpus2 , self.token2idx, self.idx2token = cg.get_cleaned_corpus()
        self.max_length = max_length
        self.random_sample = random_sample

    def __len__(self):
        return len(self.corpus1)

    def __getitem__(self, idx):
        song = self.corpus1[idx]
        if self.random_sample:
            prefix_len = random.randint(2, min(len(song), self.max_length))
            song = song[:prefix_len]
        return torch.tensor(song, dtype=torch.long)


def collate_fn(batch):
    """
    Prepares packed input and target sequences for training.
    """
    if len(batch) == 0:
        return None, None
    inputs = [x[:-1] for x in batch]
    targets = [x[1:] for x in batch]
    packed_inputs = pack_sequence(inputs, enforce_sorted=False)
    packed_targets = pack_sequence(targets, enforce_sorted=False)
    return packed_inputs, packed_targets


class LyricWriter(nn.Module):

    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder.weight = self.word_embedding.weight  # Weight tying

    def forward(self, packed_input: PackedSequence):
        embedded = self.word_embedding(packed_input.data)
        packed_embedded = PackedSequence(
            embedded, packed_input.batch_sizes,
            packed_input.sorted_indices, packed_input.unsorted_indices
        )
        packed_output, _ = self.gru(packed_embedded)
        return self.decoder(packed_output.data)


def train_nn(
        epochs=5, batch_size=16, lr=1e-3,
        max_length_final=20, load_model=True,
        model_path="model.pt"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Initializing corpus 1...")
    dataset = SongData()
    print("Finished initializing.")
    # vocab size is len + 1 because when generating dictionaries '%' character
    # got assigned 2998 index while it was already the 12th index.

    model = LyricWriter(vocab_size=len(dataset.token2idx) + 1).to(device)

    if load_model:
        model.load_state_dict(torch.load(model_path, map_location=device))
        dataset.random_sample = False
        dataset.max_length = max_length_final

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_batches = len(dataset) // batch_size
    increment_steps = max(1, total_batches // max(1, max_length_final - dataset.max_length))


    for epoch in range(epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")


        for i, (packed_inputs, packed_targets) in enumerate(pbar):

            packed_inputs = PackedSequence(
                packed_inputs.data,
                packed_inputs.batch_sizes,
                packed_inputs.sorted_indices,
                packed_inputs.unsorted_indices
            )
            packed_targets = PackedSequence(
                packed_targets.data,
                packed_targets.batch_sizes,
                packed_targets.sorted_indices,
                packed_targets.unsorted_indices
            )
            packed_inputs = packed_inputs.to(device)
            packed_targets = packed_targets.to(device)

            model.train()
            optimizer.zero_grad()

            logits = model(packed_inputs)
            loss = F.cross_entropy(logits, packed_targets.data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix({'loss': loss.item(), 'max_length': dataset.max_length})

            # Increase sequence length as training progresses
            if (i + 1) % increment_steps == 0 and dataset.max_length < max_length_final:
                dataset.max_length += 1
            if dataset.max_length >= max_length_final:
                dataset.random_sample = False
        # Save model after each epoch
        torch.save(model.state_dict(), model_path)
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
    print("Training complete.")


def predict_next_string(model, prefix, token2idx, idx2token, max_len=100, temperature=0.25):
    """
    Generates a continuation of the input string using the trained model.
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = cg.tokenize('^' + prefix, token2idx)
    generated = tokens[:]

    while len(generated) < max_len:
        input_tensor = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        packed_input = pack_sequence([input_tensor.squeeze(0)], enforce_sorted=False)

        with torch.no_grad():
            logits = model(packed_input)[-1]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token_id)

        if idx2token[next_token_id] == '%':
            break

    return ''.join([idx2token[tok] for tok in generated])


def complete_string(prefix, model_path="model.pt", temperature=0.25):
    """
    Loads the model and generates a completion for the given prefix.
    """
    token2idx, idx2token = cg.get_dictionaries()
    model = LyricWriter(vocab_size=len(token2idx) + 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return predict_next_string(model, prefix, token2idx, idx2token, temperature=temperature)


train_nn(epochs=3, lr=0.001)
# print(complete_string("all this time we've been together\n(Oh-oh-oh) and I still don't know how you feel\n", temperature=0.2))
