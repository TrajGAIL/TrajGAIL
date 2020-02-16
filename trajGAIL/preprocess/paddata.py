import torch
from torch.utils.data import Dataset, DataLoader

"""
We are going to use the Dataset interface provided
by pytorch which is really convenient when it comes to
batching our data
"""
class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
        seq_tensor, target_tensor, seq_lengths, raw_data
    """

    def __init__(self, states_tensor, actions_tensor, length_tensor):
        assert states_tensor.size(0) == length_tensor.size(0) == actions_tensor.size(0)
        self.states_tensor = states_tensor
        self.actions_tensor = actions_tensor
        self.length_tensor = length_tensor # used for variable-sized sequences

    def __getitem__(self, index):
        return self.states_tensor[index], self.actions_tensor[index], self.length_tensor[index]

    def __len__(self):
        return self.states_tensor.size(0)


# indexing data to a dictionary

def pad_sequences_states(seqs, seq_lengths, state_length=125):
    seq_tensor = torch.zeros((len(seqs), seq_lengths.max(), state_length)).float()
    for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
        seq_tensor[idx, :seqlen, :state_length] = torch.FloatTensor(seq).view(-1, 1)
    return seq_tensor

def pad_sequences_actions(seqs, seq_lengths):
    seq_tensor = torch.zeros((len(seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor

def create_dataset(data_states, data_actions, state_length=125, bs=4):
    seq_lengths = torch.LongTensor([len(s) for s in data_states])
    seq_state_tensor = pad_sequences_states(data_states, seq_lengths, state_length) 
    seq_action_tensor = pad_sequences_actions(data_actions, seq_lengths)
    return DataLoader(PaddedTensorDataset(seq_state_tensor, seq_action_tensor, seq_lengths), batch_size=bs)
