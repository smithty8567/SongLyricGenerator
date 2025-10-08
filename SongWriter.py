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

    def __init__(self, max_length=4, item_size = 3):
        self.corpus , self.token2idx, self.idx2token = cg.get_cleaned_corpus()
        self.max_length = max_length
        self.item_size = item_size

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        end = min(idx+self.item_size, self.__len__())
        song_lines = []
        lines = self.corpus[idx: end]
        for line in lines:
            song_lines.extend(line)
        return torch.tensor(song_lines, dtype=torch.long)

    def updateSongs(self, song_amt):
        self.corpus = cg.grab_new_songs(num_songs = song_amt)


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


class SongWriter(nn.Module):

    def __init__(self, vocab_size, hidden_size=200):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3, bidirectional = False)
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
        epochs=5, batch_size=32, lr=1e-3,
        max_length_final=20, load_model=True,
        model_path="model.pt", num_songs = 125000
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    dataset = SongData()

    model = SongWriter(vocab_size=len(dataset.token2idx)).to(device)

    if load_model:
        print("previous model loaded")
        model.load_state_dict(torch.load(model_path, map_location=device))
        dataset.random_sample = True
        dataset.max_length = max_length_final

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_batches = len(dataset) // batch_size
    increment_steps = max(1, total_batches // max(1, max_length_final - dataset.max_length))

    for epoch in range(epochs):
        dataset.updateSongs(song_amt = num_songs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=8)
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

            # Increase sequence length as training progresses
            if (i + 1) % increment_steps == 0 and dataset.max_length < max_length_final:
                dataset.max_length += 1
            if dataset.max_length >= max_length_final:
                dataset.random_sample = False
        # Save model after each epoch
        torch.save(model.state_dict(), model_path)
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1} loss: {loss.item()}")
    print("Training complete.")


def predict_next_string(model, prefix, token2idx, idx2token, max_len=30, temperature=0.15):

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


def complete_string(prefix, model_path="model.pt", temperature=0.15):

    token2idx, idx2token = cg.get_dictionaries()
    model = SongWriter(vocab_size=len(token2idx))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return predict_next_string(model, prefix, token2idx, idx2token, temperature=temperature)

if __name__ == "__main__":
    train_nn(epochs = 10)
