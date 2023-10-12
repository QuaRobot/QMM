import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    """wrap in PyTorch Dataset"""
    def __init__(self, examples):
        """

        :param examples: examples returned by VUA_All_Processor or Verb_Processor
        """
        super(dataset, self).__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def collate_fn(examples):
    ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels = map(list, zip(*examples))

    ids_r = torch.tensor(ids_r, dtype=torch.long)
    att_r = torch.tensor(att_r, dtype=torch.long)
    tar_r = torch.tensor(tar_r, dtype=torch.long)

    ids_rg = torch.tensor(ids_rg, dtype=torch.long)
    att_rg = torch.tensor(att_rg, dtype=torch.long)

    # ids_i = torch.tensor(ids_i, dtype=torch.long)
    # att_i = torch.tensor(att_i, dtype=torch.long)
    # tar_i = torch.tensor(tar_i, dtype=torch.long)
    # ids_ig = torch.tensor(ids_ig, dtype=torch.long)
    # att_ig = torch.tensor(att_ig, dtype=torch.long)
    # pos_i = torch.tensor(pos_i, dtype=torch.long)
    # pos_a = torch.tensor(pos_a, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.long)

    gloss_att = torch.tensor(gloss_att, dtype=torch.long)

    labels = torch.tensor(labels, dtype=torch.long)

    return ids_r, att_r, tar_r, ids_rg, att_rg, pos, gloss_att, labels
