import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
import transformers
from metalloscribe.model import EncoderDecoderModel
from metalloscribe.dataset import CustomDataset, custom_collate_fn
import argparse
import math
import os
import json
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from metalloscribe.utils import count_atom_accuracy, calculate_positional_accuracy, find_atoms_with_valid_xyz
from metalloscribe.tokenizer import PAD_ID
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch.distributed as dist
from torch.utils.data import random_split
from transformers import get_scheduler

def get_args(notebook=False):   
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--fast_dev_run', nargs='?', const=False, type=int, default=False)

    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--images_folder', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--use_training_for_validation', action='store_true')
    parser.add_argument('--train_validation_ratio', type=float, default=0.8)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--format', type=str, default='reaction')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)

    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--eval_per_epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_train_example', type=int, default=None)
    parser.add_argument('--limit_train_batches', type=int, default=None)
    parser.add_argument('--limit_val_batches', type=int, default=None)

    parser.add_argument('--roberta_checkpoint', type=str, default = "roberta-base")

    parser.add_argument('--corpus', type=str, default = "chemu")

    parser.add_argument('--cache_dir')

    parser.add_argument('--eval_truncated', action='store_true')

    parser.add_argument('--max_seq_length', type = int, default=512)

    args = parser.parse_args([]) if notebook else parser.parse_args()

    return args

class LitEncoderDecoderModel(LightningModule):
    def __init__(self, args):
        super(LitEncoderDecoderModel, self).__init__()
        self.model = EncoderDecoderModel()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
        self.lr = args.lr
        self.vocab_size = self.model.vocab_size
        self.args = args

        #self.model = build_model(args)

        self.validation_step_outputs = []

    def forward(self, images, labels=None):
        return self.model(images, labels)

    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']
        outputs = self(images, labels)
        loss = self.criterion(outputs.view(-1, self.vocab_size+300), labels.view(-1))
        #print(f"outputs.view(-1) {outputs.view(-1, self.vocab_size+300)}")
        #print(f"Labels.view(-1) {labels.view(-1)}")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']

        # Perform inference
        sequence = self.model.inference(images)

        # Calculate metrics
        all_overall_atom_error, all_hydrogen_error, all_carbon_error, all_metal_error, all_other_error = count_atom_accuracy(labels, sequence)
        #, all_label_counts, all_prediction_counts 
        valid_xyz_percentages = find_atoms_with_valid_xyz(labels, sequence)

        # Accumulate metrics for this batch
        self.validation_step_outputs.append({
            'all_overall_atom_error': all_overall_atom_error.to("cpu"),
            'all_hydrogen_error': all_hydrogen_error.to("cpu"),
            'all_carbon_error': all_carbon_error.to("cpu"),
            'all_metal_error': all_metal_error.to("cpu"),
            'all_other_error': all_other_error.to("cpu"),
            #'all_label_counts': all_label_counts,
            #'all_prediction_counts': all_prediction_counts,
            'valid_xyz_percentages': valid_xyz_percentages.to("cpu"),
        })

        # Explicitly delete the batch and clear cache to free GPU memory
        del images, labels, sequence
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # Initialize accumulators
        accumulated_scores = {
            'all_overall_atom_error': 0,
            'all_hydrogen_error': 0,
            'all_carbon_error': 0,
            'all_metal_error': 0,
            'all_other_error': 0,
            #'all_label_counts': np.zeros(123),
            #'all_prediction_counts': np.zeros(123),
            'valid_xyz_percentages': 0
        }
        
        if self.trainer.is_global_zero:
            for output in self.validation_step_outputs:
                for i in range(len(output['all_overall_atom_error'])):
                    #for atom in range(5, 123):
                    #    accumulated_scores['all_label_counts'][atom] += output['all_label_counts'][i][atom]
                    #    accumulated_scores['all_prediction_counts'][atom] += output['all_prediction_counts'][i][atom]
                    
                    accumulated_scores['all_overall_atom_error'] += output['all_overall_atom_error'][i]
                    accumulated_scores['all_hydrogen_error'] += output['all_hydrogen_error'][i]
                    accumulated_scores['all_carbon_error'] += output['all_carbon_error'][i]
                    accumulated_scores['all_metal_error'] += output['all_metal_error'][i]
                    accumulated_scores['all_other_error'] += output['all_other_error'][i]
                    accumulated_scores['valid_xyz_percentages'] += output['valid_xyz_percentages'][i]

        # Gather the accumulated scores across all processes
        if self.trainer.num_devices > 1:
            gathered_scores = [None for _ in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_scores, accumulated_scores)
            # Combine results from all processes
            combined_scores = {
                'all_overall_atom_error': 0,
                'all_hydrogen_error': 0,
                'all_carbon_error': 0,
                'all_metal_error': 0,
                'all_other_error': 0,
                #'all_label_counts': np.zeros(123),
                #'all_prediction_counts': np.zeros(123),
                'valid_xyz_percentages': 0
            }
            for score in gathered_scores:
                combined_scores['all_overall_atom_error'] += score['all_overall_atom_error']
                combined_scores['all_hydrogen_error'] += score['all_hydrogen_error']
                combined_scores['all_carbon_error'] += score['all_carbon_error']
                combined_scores['all_metal_error'] += score['all_metal_error']
                combined_scores['all_other_error'] += score['all_other_error']
                combined_scores['valid_xyz_percentages'] += score['valid_xyz_percentages']
                #for atom in range(5, 123):
                #    combined_scores['all_label_counts'][atom] += score['all_label_counts'][atom]
                #    combined_scores['all_prediction_counts'][atom] += score['all_prediction_counts'][atom]
            accumulated_scores = combined_scores

        # Aggregate the final scores
        total_entries= sum(len(d['all_overall_atom_error']) for d in self.validation_step_outputs)
        final_scores = {
            'all_overall_atom_error': accumulated_scores['all_overall_atom_error'] / total_entries,
            'all_hydrogen_error': accumulated_scores['all_hydrogen_error'] / total_entries,
            'all_carbon_error': accumulated_scores['all_carbon_error'] / total_entries,
            'all_metal_error': accumulated_scores['all_metal_error'] / total_entries,
            'all_other_error': accumulated_scores['all_other_error'] / total_entries,
            'valid_xyz_percentages': accumulated_scores['valid_xyz_percentages'] / total_entries,
        }
        
        # Calculate and log average prediction accuracy errors
        avg_prediction_accuracy_error = {}
        #for atom in range(5, 123):
        #    if accumulated_scores['all_label_counts'][atom] > 0:
        #        avg_prediction_accuracy_error[atom] = abs(100 - (accumulated_scores['all_prediction_counts'][atom] / accumulated_scores['all_label_counts'][atom]) * 100)
        #        self.log(f'val/avg_prediction_accuracy_error_{atom}', avg_prediction_accuracy_error[atom], prog_bar=True, rank_zero_only=True, sync_dist=True)

        self.log('val/all_overall_atom_error', final_scores['all_overall_atom_error'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/all_hydrogen_error', final_scores['all_hydrogen_error'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/all_carbon_error', final_scores['all_carbon_error'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/all_metal_error', final_scores['all_metal_error'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/all_other_error', final_scores['all_other_error'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/avg_valid_xyz_percentage', final_scores['valid_xyz_percentages'], prog_bar=True, rank_zero_only=True, sync_dist=True)

        # Clear the validation outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class CustomDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.collate_fn = custom_collate_fn

    def prepare_data(self):
        
        args = self.args
        if args.do_train:
            if args.use_training_for_validation:
                # Split the training dataset into training and validation subsets
                self.train_dataset=CustomDataset(args)
                total_size = len(self.train_dataset)
                train_size = int(total_size * args.train_validation_ratio)
                val_size = total_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])
            else:
                self.train_dataset = CustomDataset(args, split='train') # split not implemented yet
        else:
            if self.args.do_train or self.args.do_valid:
                self.train_dataset=CustomDataset(args)
                total_size = len(self.train_dataset)
                train_size = int(total_size * args.train_validation_ratio)
                val_size = total_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])
        
        if self.args.do_test:
            self.test_dataset = CustomDataset(args, split='valid') # split not implemented yet
        '''
        args = self.args
        if args.do_train:
            if args.use_training_for_validation:
                # Split the training dataset into training and validation subsets
                full_dataset = CustomDataset(args)
                self.train_dataset = SingleExampleDataset(args)
                self.val_dataset = SingleExampleDataset(args)
            else:
                self.train_dataset = SingleExampleDataset(args)  # Using single example for training
        else:
            if self.args.do_train or self.args.do_valid:
                self.val_dataset = SingleExampleDataset(args)  # Using single example for validation
        
        if self.args.do_test:
            self.test_dataset = SingleExampleDataset(args)  # Using single example for testing
        '''
    def print_stats(self):
        if self.args.do_train:
            print(f'Train dataset: {len(self.train_dataset)}')
        if self.args.do_train or self.args.do_valid:
            print(f'Valid dataset: {len(self.val_dataset)}')
        if self.args.do_test:
            print(f'Test dataset: {len(self.test_dataset)}')

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn) # Shuffle = true?

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)
        return filepath 

def main():
    torch.set_printoptions(edgeitems=5)

    transformers.utils.logging.set_verbosity_error()
    args = get_args()

    pl.seed_everything(args.seed, workers = True)

    if args.do_train:
        model = LitEncoderDecoderModel(args)
    else:
        model = LitEncoderDecoderModel.load_from_checkpoint(os.path.join(args.save_path, 'checkpoints/best.ckpt'), strict=False,
                                        args=args)

    dm = CustomDataModule(args)
    dm.prepare_data()
    #dm.print_stats()

    val_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.save_path, 'checkpoints'), monitor='val/all_overall_atom_error',
                                 mode='min', save_top_k=1, filename='best', save_last=True)
    train_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, 'checkpoints'),
        monitor='train_loss',
        mode='min',
        save_top_k=-1,
        filename='best_train',
        save_last=True,
        every_n_epochs=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(
        name='',  # You can give your run a name here
        save_dir=args.save_path,  # Directory where the wandb logs will be saved
        project='MetalloScribe'  # Replace with your project name in wandb
    )

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator='gpu',
        precision = 16,
        devices=args.gpus,
        #devices=1,
        logger=wandb_logger,
        default_root_dir=args.save_path,
        callbacks=[val_checkpoint, train_checkpoint, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=5,
        deterministic='warn',
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches)

    if args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(args.save_path, 'checkpoints/last.ckpt') if args.resume else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        #print(val_checkpoint.best_model_path)
        #print(val_checkpoint.best_model_score)
        model = LitEncoderDecoderModel.load_from_checkpoint(val_checkpoint.best_model_path, args=args)

    if args.do_valid:

        model.eval_dataset = dm.val_dataset

        trainer.validate(model, datamodule=dm)

    if args.do_test:

        model.test_dataset = dm.test_dataset

        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()