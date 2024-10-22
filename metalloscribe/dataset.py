import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import NodeTokenizer
import json
from transformers import AutoFeatureExtractor
from .tokenizer import SOS_ID, EOS_ID, PAD_ID, MASK_ID

# Tokenizer args
args = type('', (), {})()  # Create a simple object to hold attributes
args.formats = ['atomtok_coords']
args.vocab_file = '../vocab/vocab_periodic_table.json'
args.coord_bins = 100  # Randomly chosen, number of bins we split the image into
args.sep_xyz = True
args.continuous_coords = False  # Should be generally set to False
args.debug = False

# Load NodeTokenizer
tokenizer = NodeTokenizer(input_size=args.coord_bins, path=args.vocab_file, sep_xyz=args.sep_xyz, continuous_coords=args.continuous_coords,
                          debug=args.debug)
model_name = "microsoft/swin-base-patch4-window12-384-in22k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    identifiers = [item['ID'] for item in data]
    label_sequences = [item['Molfile'] for item in data]
    return identifiers, label_sequences

class CustomDataset(Dataset):
    def __init__(self, args):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.images_folder = args.images_folder

        self.identifiers, self.label_sequences = self.load_and_filter_data(args.train_file)
        
    def load_and_filter_data(self, train_file):
        identifiers, label_sequences = load_data(train_file)
        filtered_identifiers = []
        filtered_label_sequences = []

        for identifier, label_sequence in zip(identifiers, label_sequences):
            labels, _ = self.tokenizer.molfile_to_sequence(label_sequence) # indices not used
            if len(labels) <= 1024: # So that we don't have sequences too long
                filtered_identifiers.append(identifier)
                filtered_label_sequences.append(label_sequence)

        return filtered_identifiers, filtered_label_sequences

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        identifier = self.identifiers[idx]
        label_sequence = self.label_sequences[idx]

        # Load and process image
        image = Image.open(os.path.join(self.images_folder, f'{identifier}.png')).convert('RGB')
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()

        # Process label sequence
        labels, _ = self.tokenizer.molfile_to_sequence(label_sequence) # indices not used

        labels = torch.tensor(labels, dtype=torch.long) # convert to tensor

        return {'images': image, 'labels': labels}


def custom_collate_fn(batch):
    images = [item['images'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad labels
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PAD_ID)

    # Stack images
    images = torch.stack(images)

    return {'images': images, 'labels': padded_labels}

'''
class SingleExampleDataset(CustomDataset):
    def __init__(self, args):
        super().__init__(args)
        self.example = super().__getitem__(0)  # Initialize self.example using the parent class method

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.example
'''
