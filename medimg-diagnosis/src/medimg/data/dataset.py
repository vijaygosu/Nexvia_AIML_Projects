import csv

import torch
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, manifest: str, split: str, labels: list, transform):
        self.labels, self.transform = labels, transform
        with open(manifest, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split '{split}' in {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.transform(Image.open(r["image_path"]).convert("L"))
        y = torch.tensor([float(r[l]) for l in self.labels])
        return x, y, i

    def targets(self):
        return torch.tensor([[float(r[l]) for l in self.labels] for r in self.rows])
