import os
import json
import random
import numpy as np
import torch
from SmilesPE.pretokenizer import atomwise_tokenizer

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
MASK = '<mask>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4


class Tokenizer(object):

    def __init__(self, path=None):
        self.stoi = {} # This dictionary maps each token (a string) to a unique integer ID.
        self.itos = {} # This dictionary maps each integer ID back to its corresponding token (a string).
        if path:
            self.load(path) # Specify the file path for saving or loading the vocabulary (the mappings between tokens and their IDs).

    def __len__(self): # Size of dictionary
        return len(self.stoi)

    @property
    def output_constraint(self): # Output can be anything
        return False

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.stoi, f) # Save your stoi dictionary

    def load(self, path):
        with open(path) as f:
            self.stoi = json.load(f) # Load stoi dictionary
        self.itos = {item[1]: item[0] for item in self.stoi.items()} # Create itos dictionary from stoi dictionary

    def fit_on_texts(self, texts): # Generate vocabulary for stoi dict and itos dict
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = [PAD, SOS, EOS, UNK] + list(vocab)
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID

    def text_to_sequence(self, text, tokenized=True): # Convert text to a tokenized sequence
        sequence = []
        sequence.append(self.stoi['<sos>'])
        if tokenized:
            tokens = text.split(' ')
        else:
            tokens = atomwise_tokenizer(text)
        for s in tokens:
            if s not in self.stoi:
                s = '<unk>'
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts): # Converts a list of texts to a list of tokenized sequences
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence): # Converts a tokenized sequence back to text
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences): # Converts a list of tokenized sequences back to a list of texts
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

    def sequence_to_smiles(self, sequence):
        return {'smiles': self.predict_caption(sequence)}


class NodeTokenizer(Tokenizer):

    def __init__(self, input_size=100, path=None, sep_xyz=False, continuous_coords=False, debug=False):
        super().__init__(path)
        self.maxx = input_size  # height
        self.maxy = input_size  # width
        self.maxz = input_size  # depth
        self.sep_xyz = sep_xyz
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK] # Padding, Start of sequence, End of sequence, Unknown, Mask
        self.continuous_coords = continuous_coords # Whether coordinates of atoms continuous or discrete
        self.debug = debug

    def __len__(self): # Size of dictionary
        if self.sep_xyz: # Whether x and y values are tokenized to different ranges
            return self.offset + self.maxx + self.maxy + self.maxz
        else:
            return self.offset + max(self.maxx, self.maxy, self.maxz)

    @property
    def offset(self): # Ensure coordinates and characters don't have same tokenized representation
        return len(self.stoi) 

    @property
    def output_constraint(self): # There are output constraints if coordinates are discretized
        return not self.continuous_coords

    def len_symbols(self): # Method to return length of dictionary
        return len(self.stoi)

    def fit_atom_symbols(self, atoms): # Add atom symbols to list of special tokens
        vocab = self.special_tokens + list(set(atoms))
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def is_x(self, x): # Check if a token is an "x-coordinate" token
        return self.offset <= x < self.offset + self.maxx

    def is_y(self, y): # Check if a token is an "y-coordinate" token
        if self.sep_xyz:
            return self.offset + self.maxx <= y
        return self.offset <= y

    def is_z(self, z): # Check if a token is an "z-coordinate" token
        if self.sep_xyz:
            return self.offset + self.maxx + self.maxy <= z
        return self.offset <= z

    def is_symbol(self, s): # Check if a token is a "synbol" token (includes atoms and other symbols found)
        return len(self.special_tokens) <= s < self.offset or s == UNK_ID

    def is_atom(self, id): # Check if a token is an "atom" token
        if self.is_symbol(id):
            return self.is_atom_token(self.itos[id])
        return False

    def is_atom_token(self, token): # Judging if a token is an atom token
        return token.isalpha() or token.startswith("[") or token == '*' or token == UNK

    def x_to_id(self, x): # Tokenize an x coordinate by discretizing it
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y): # Tokenize a y coordinate by discretizing it
        if self.sep_xyz:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))
    
    def z_to_id(self, z): # Tokenize a z coordinate by discretizing it
        if self.sep_xyz:
            return self.offset + self.maxx + self.maxy + round(z * (self.maxz - 1))
        return self.offset + round(z * (self.maxz - 1))

    def id_to_x(self, id): # Untokenize an x coordinate (approximately)
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id): # Untokenize a y coordinate (approximately)
        if self.sep_xyz:
            return (id - self.offset - self.maxx) / (self.maxy - 1)
        return (id - self.offset) / (self.maxy - 1)
    
    def id_to_z(self, id): # Untokenize a z coordinate (approximately)
        if self.sep_xyz:
            return (id - self.offset - self.maxx - self.maxy) / (self.maxz - 1)
        return (id - self.offset) / (self.maxz - 1)
    
    def get_output_mask(self, id): # Judging what characters can be represented by a given token ID
        mask = [False] * len(self)
        if self.continuous_coords:
            return mask
        if self.is_atom(id):
            return [True] * self.offset + [False] * self.maxx + [True] * (self.maxy + self.maxz)
        if self.is_x(id):
            return [True] * (self.offset + self.maxx) + [False] * self.maxy + [True] * self.maxz
        if self.is_y(id):
            return [True] * (self.offset + self.maxx + self.maxy) + [False] * self.maxz
        if self.is_z(id):
            return [False] * self.offset + [True] * (self.maxx + self.maxy + self.maxz)
        return mask

    def symbol_to_id(self, symbol): # Symbol to ID
        if symbol not in self.stoi:
            return UNK_ID
        return self.stoi[symbol]

    def symbols_to_labels(self, symbols): # Symbols to IDs
        labels = []
        for symbol in symbols:
            labels.append(self.symbol_to_id(symbol))
        return labels

    def labels_to_symbols(self, labels): #IDs to Symbols
        symbols = []
        for label in labels:
            symbols.append(self.itos[label])
        return symbols


    '''
    Everything below this not equiped to handle 3D except molfile stuff
    '''

    def nodes_to_grid(self, nodes): # Returns a 2D list, where each position has a symbol ID if there's an atom there in the image
        coords, symbols = nodes['coords'], nodes['symbols']
        grid = np.zeros((self.maxx, self.maxy), dtype=int)
        for [x, y], symbol in zip(coords, symbols):
            x = round(x * (self.maxx - 1))
            y = round(y * (self.maxy - 1))
            grid[x][y] = self.symbol_to_id(symbol)
        return grid

    def grid_to_nodes(self, grid): # Takes in a grid and outputs the corresponding dictionary of symbols and their coordinates/indices
        coords, symbols, indices = [], [], []
        for i in range(self.maxx):
            for j in range(self.maxy):
                if grid[i][j] != 0:
                    x = i / (self.maxx - 1)
                    y = j / (self.maxy - 1)
                    coords.append([x, y])
                    symbols.append(self.itos[grid[i][j]])
                    indices.append([i, j])
        return {'coords': coords, 'symbols': symbols, 'indices': indices}

    def nodes_to_sequence(self, nodes): # Takes in a dictionary of symbols and coordinates and outputs a sequence
        coords, symbols = nodes['coords'], nodes['symbols']
        labels = [SOS_ID]
        for (x, y), symbol in zip(coords, symbols):
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            labels.append(self.x_to_id(x))
            labels.append(self.y_to_id(y))
            labels.append(self.symbol_to_id(symbol))
        labels.append(EOS_ID)
        return labels

    def sequence_to_nodes(self, sequence): # Takes in a sequence and outputs the corresponding dictionary
        coords, symbols = [], []
        i = 0
        if sequence[0] == SOS_ID:
            i += 1
        while i + 2 < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if self.is_x(sequence[i]) and self.is_y(sequence[i+1]) and self.is_symbol(sequence[i+2]):
                x = self.id_to_x(sequence[i])
                y = self.id_to_y(sequence[i+1])
                symbol = self.itos[sequence[i+2]]
                coords.append([x, y])
                symbols.append(symbol)
            i += 3
        return {'coords': coords, 'symbols': symbols}

    def extract_coordinates_and_symbols(self, molfile): # Extract atoms and respective coordinates from molfile
        lines = molfile.split('\n')
        header_end_index = 4  # Assuming the first four lines are header
        coordinates_and_symbols = []

        for line in lines[header_end_index:]:
            parts = line.split()
            if len(parts) < 4:
                break  # End of the coordinates and symbols section
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                symbol = parts[3]
                coordinates_and_symbols.append((x, y, z, symbol))
            except ValueError:
                break  # Reached a line that does not match the expected format

        return coordinates_and_symbols
    
    def molfile_to_sequence(self, molfile, mask_ratio=0, atom_only=False):
        # Extract atoms and respective coordinates from molfile
        lines = molfile.split('\n')
        header_end_index = 4  # Assuming the first four lines are header
        coordinates_and_symbols = []

        xdim_max, ydim_max, zdim_max = -1e9, -1e9, -1e9
        xdim_min, ydim_min, zdim_min = 1e9, 1e9, 1e9

        for line in lines[header_end_index:]:
            parts = line.split()
            if len(parts) < 4:
                break  # End of the coordinates and symbols section
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                xdim_max = max(xdim_max, x)
                ydim_max = max(ydim_max, y)
                zdim_max = max(zdim_max, z)
                xdim_min = min(xdim_min, x)
                ydim_min = min(ydim_min, y)
                zdim_min = min(zdim_min, z)
                symbol = parts[3]
                coordinates_and_symbols.append((x, y, z, symbol))
            except ValueError:
                break  # Reached a line that does not match the expected format

        coordinates_and_symbols_normalized = []
        for (x,y,z,symbol) in coordinates_and_symbols:
            coordinates_and_symbols_normalized.append(((x-xdim_min)/max(xdim_max-xdim_min, 1e-6), (y-ydim_min)/max(ydim_max-ydim_min, 1e-6), 
                                                       (z-zdim_min)/max(zdim_max-zdim_min, 1e-6), symbol))

        labels = [SOS_ID]
        indices = []
        atom_idx = -1
        
        for (x, y, z, symbol) in coordinates_and_symbols_normalized:
            if atom_only and not self.is_atom_token(symbol):
                continue
            if symbol in self.stoi:
                labels.append(self.stoi[symbol])
            else:
                if self.debug:
                    print(f'{symbol} not in vocab')
                labels.append(UNK_ID)
            if self.is_atom_token(symbol):
                atom_idx += 1
                if not self.continuous_coords:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        labels.append(MASK_ID)
                        labels.append(MASK_ID)
                        labels.append(MASK_ID)
                    else:
                        assert 0 <= x <= 1
                        assert 0 <= y <= 1
                        assert 0 <= z <= 1
                        labels.append(self.x_to_id(x))
                        labels.append(self.y_to_id(y))
                        labels.append(self.z_to_id(z))
                indices.append(len(labels) - 1)
        labels.append(EOS_ID)

        return labels, indices


    def smiles_to_sequence(self, smiles, coords=None, mask_ratio=0, atom_only=False):
        tokens = atomwise_tokenizer(smiles)
        labels = [SOS_ID]
        indices = []
        atom_idx = -1
        for token in tokens:
            if atom_only and not self.is_atom_token(token):
                continue
            if token in self.stoi:
                labels.append(self.stoi[token])
            else:
                if self.debug:
                    print(f'{token} not in vocab')
                labels.append(UNK_ID)
            if self.is_atom_token(token):
                atom_idx += 1
                if not self.continuous_coords:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        labels.append(MASK_ID)
                        labels.append(MASK_ID)
                    elif coords is not None:
                        if atom_idx < len(coords):
                            x, y = coords[atom_idx]
                            assert 0 <= x <= 1
                            assert 0 <= y <= 1
                        else:
                            x = random.random()
                            y = random.random()
                        labels.append(self.x_to_id(x))
                        labels.append(self.y_to_id(y))
                indices.append(len(labels) - 1)
        labels.append(EOS_ID)
        return labels, indices

    def sequence_to_smiles(self, sequence):
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices = [], [], []
        for i, label in enumerate(sequence):
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                continue
            token = self.itos[label]
            smiles += token
            if self.is_atom_token(token):
                if has_coords:
                    if i+3 < len(sequence) and self.is_x(sequence[i+1]) and self.is_y(sequence[i+2]):
                        x = self.id_to_x(sequence[i+1])
                        y = self.id_to_y(sequence[i+2])
                        coords.append([x, y])
                        symbols.append(token)
                        indices.append(i+3)
                else:
                    if i+1 < len(sequence):
                        symbols.append(token)
                        indices.append(i+1)
        results = {'smiles': smiles, 'symbols': symbols, 'indices': indices}
        if has_coords:
            results['coords'] = coords
        return results


def get_tokenizer(args):
    tokenizer = {}
    for format_ in args.formats:
        if format_ == 'atomtok':
            if args.vocab_file is None:
                args.vocab_file = os.path.join(os.path.dirname(__file__), 'vocab/vocab_periodic_table.json')
            tokenizer['atomtok'] = Tokenizer(args.vocab_file)
        elif format_ == "atomtok_coords":
            if args.vocab_file is None:
                args.vocab_file = os.path.join(os.path.dirname(__file__), 'vocab/vocab_periodic_table.json')
            tokenizer["atomtok_coords"] = NodeTokenizer(args.coord_bins, args.vocab_file, args.sep_xyz,
                                                        continuous_coords=args.continuous_coords)
        
    return tokenizer