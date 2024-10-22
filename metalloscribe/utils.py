import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def isMetal(x):
    # List of atomic numbers corresponding to metals
    metals_atomic_numbers = [
        # Alkali metals
        3, 11, 19, 37, 55, 87,  
        # Alkaline earth metals
        4, 12, 20, 38, 56, 88,
        # Transition metals
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        72, 73, 74, 75, 76, 77, 78, 79, 80,
        104, 105, 106, 107, 108, 109, 110, 111, 112,
        # Lanthanides
        57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        # Actinides
        89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
        # Other metals
        13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116
    ]
    
    # Adjust for input range
    atomic_number = x - 4  # since H is 1 and given as 5

    if atomic_number in metals_atomic_numbers:
        return True
    else:
        return False

def count_atom_accuracy(labels, predictions):
    # Count occurrences of each atom type in labels and predictions
    all_overall_error = []
    all_hydrogen_error = []
    all_carbon_error = []
    all_metal_error = []
    all_other_error = []
    #all_label_counts = []
    #all_prediction_counts = []
    
    #print(f"Labels shape: {labels.shape}")
    #print(f"Predictions shape: {predictions.shape}")

    for label, prediction in zip(labels, predictions):
        # Initialize counts for atoms 5 through 122
        label_counts = {i: 0 for i in range(5, 123)}
        prediction_counts = {i: 0 for i in range(5, 123)}

        label = label.clone().detach()
        prediction = prediction.clone().detach()

        unique_labels, label_occurrences = torch.unique(label, return_counts=True)
        unique_predictions, prediction_occurrences = torch.unique(prediction, return_counts=True)

        for atom, count in zip(unique_labels, label_occurrences):
            if 5 <= atom.item() <= 122:
                label_counts[atom.item()] += count.item()
        
        for atom, count in zip(unique_predictions, prediction_occurrences):
            if 5 <= atom.item() <= 122:
                prediction_counts[atom.item()] += count.item()

        # Calculate percentage error for each atom type present in labels
        errors = []
        for atom in range(5, 123):
            if label_counts[atom] > 0:
                label_count = label_counts[atom]
                prediction_count = prediction_counts[atom]
                percentage_error = abs(label_count - prediction_count) / label_count * 100
                errors.append(percentage_error)
    
        # Calculate overall accuracy as the average percentage error
        overall_error = (sum(errors) / len(errors) if errors else 0)
        hydrogen_error = (abs(label_counts[5] - prediction_counts[5]) / label_counts[5] * 100 if label_counts[5] > 0 else 0)
        carbon_error = (abs(label_counts[10] - prediction_counts[10]) / label_counts[10] * 100 if label_counts[10] > 0 else 0)
        metal_error = sum(abs(label_counts[atom] - prediction_counts[atom]) / label_counts[atom] * 100 for atom in range(5, 123) if label_counts[atom] > 0 and isMetal(atom))
        metal_error /= max(sum(1 for atom in range(5, 123) if label_counts[atom] > 0 and isMetal(atom)), 1)
        other_error = sum(abs(label_counts[atom] - prediction_counts[atom]) / label_counts[atom] * 100 for atom in range(5, 123) if label_counts[atom] > 0 and not isMetal(atom) and atom != 10 and atom != 5)
        other_error /= max(sum(1 for atom in range(5, 123) if label_counts[atom] > 0 and not isMetal(atom) and atom != 10 and atom != 5), 1)


        all_overall_error.append(overall_error)
        all_hydrogen_error.append(hydrogen_error)
        all_carbon_error.append(carbon_error)
        all_metal_error.append(metal_error)
        all_other_error.append(other_error)
        #all_label_counts.append(label_counts)
        #all_prediction_counts.append(prediction_counts)

    return all_overall_error, all_hydrogen_error, all_carbon_error, all_metal_error, all_other_error
#, all_label_counts, all_prediction_counts

def check_valid_xyz(tokens, start_idx):
    # Check if the next 3 indices after start_idx exist and are all >= 123
    if start_idx + 3 < len(tokens):
        return all(tokens[start_idx + i] >= 123 for i in range(1, 4))
    return False

def find_atoms_with_valid_xyz(labels, predictions):
    valid_xyz_percentages = []

    for label, prediction in zip(labels, predictions):
        valid_atom_count = 0
        total_atom_count = 0

        label = label.clone().detach()
        prediction = prediction.clone().detach()
        
        #print(f"Utils label shape: {label.shape}")
        #print(f"Utils prediction shape: {prediction.shape}")

        for i in range(len(prediction)):
            if 5 <= int(prediction[i].item()) <= 122:
                total_atom_count += 1
                if check_valid_xyz(prediction, i):
                    valid_atom_count += 1

        if total_atom_count > 0:
            valid_xyz_percentage = (valid_atom_count / total_atom_count) * 100
        else:
            valid_xyz_percentage = 0.0

        valid_xyz_percentages.append(valid_xyz_percentage)

    return valid_xyz_percentages

def extract_atoms_with_xyz(label, prediction):
    label_atoms = []
    prediction_atoms = []

    label = label.clone().detach().float()
    prediction = prediction.clone().detach().float()

    for i in range(2, len(label), 4):
        atom_type = int(label[i - 2].item())
        if 5 <= atom_type <= 122:
            xyz = label[i:i + 3].tolist()
            label_atoms.append((atom_type, xyz))

    for i in range(len(prediction)):
        if 5 <= int(prediction[i].item()) <= 122 and check_valid_xyz(prediction, i):
            atom_type = int(prediction[i].item())
            xyz = prediction[i + 1:i + 4].tolist()
            prediction_atoms.append((atom_type, xyz))

    return label_atoms, prediction_atoms

def calculate_positional_accuracy(labels, predictions):
    overall_error_percentages = []
    sorted_atom_error_percentages = []

    for label, prediction in zip(labels, predictions):
        label_atoms, prediction_atoms = extract_atoms_with_xyz(label, prediction)

        # Organize atoms by type
        label_dict = {i: [] for i in range(5, 123)}
        prediction_dict = {i: [] for i in range(5, 123)}

        for atom, xyz in label_atoms:
            label_dict[atom].append(xyz)

        for atom, xyz in prediction_atoms:
            prediction_dict[atom].append(xyz)

        overall_count = 0
        overall_matched = 0
        atom_percentage = {}

        for atom in range(5, 123):
            label_positions = np.array(label_dict[atom])
            prediction_positions = np.array(prediction_dict[atom])

            if len(label_positions) == 0:
                continue

            overall_count += len(label_positions)

            if len(prediction_positions) == 0:
                atom_percentage[atom] = 0
                continue

            # Compute distance matrix
            dist_matrix = np.linalg.norm(label_positions[:, np.newaxis, :] - prediction_positions[np.newaxis, :, :], axis=2)

            # Apply distance threshold
            dist_matrix[dist_matrix > 15] = np.inf

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched_count = sum(dist_matrix[row, col] <= 15 for row, col in zip(row_ind, col_ind))
            overall_matched += matched_count

            atom_percentage[atom] = abs(len(label_positions) - matched_count) / len(label_positions) * 100

        overall_percentage_error = abs(overall_count - overall_matched) / overall_count * 100 if overall_count > 0 else 0
        sorted_atom_percentage_error = [atom_percentage.get(atom, 0) for atom in range(5, 123)]

        overall_error_percentages.append(overall_percentage_error)
        sorted_atom_error_percentages.append(sorted_atom_percentage_error)

    return overall_error_percentages, sorted_atom_error_percentages
