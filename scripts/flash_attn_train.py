import argparse
import logging
import os
import numpy as np
from litdata import StreamingDataset, StreamingDataLoader, train_test_split
from torch import tensor, float32, long, bool, stack, Module, nn, isnan, nan_to_num, optim, cat
from flash_attn.modules.mha import MHA
from flash_attn.ops.rms_norm import RMSNorm
from flash_attn.ops.fused_dense import FusedMLP
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from torch.nn import functional as F

# Set up logging and logging, log file, and log stream
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('training.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory {path} created.")

def plot_tensorboard_scalars(ckpt_dir:str, tag=None, tags=None):
    """
    Plot scalar curves from TensorBoard logs in a given directory.

    Args:
        tag (str): Specific tag to plot.
        tags (list): List of tags to plot.
    """

    import glob
    
    # Find the most recent version
    version_dirs = glob.glob(f"{ckpt_dir}/lightning_logs/version_*")
    if not version_dirs:
        print("No training logs found.")
        return
    

    latest_version = max(version_dirs, key=lambda x: int(x.split('_')[-1]))

    # Find any event files in the directory
    files = glob.glob(os.path.join(latest_version, 'events.out.tfevents.*'))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in {latest_version}")

    # Initialize the EventAccumulator to read scalars
    ea = event_accumulator.EventAccumulator(
        latest_version,
        size_guidance={  # Load all scalar data
            event_accumulator.SCALARS: 0,
        }
    )
    ea.Reload()  # Load the event data

    # Determine which tags to plot
    available_tags = ea.Tags().get('scalars', [])
    if tags:
        plot_tags = [t for t in tags if t in available_tags]
    elif tag:
        if tag not in available_tags:
            raise ValueError(f"Tag '{tag}' not found. Available tags: {available_tags}")
        plot_tags = [tag]
    else:
        # If no tag(s) specified, plot all scalar tags
        plot_tags = available_tags

    if not plot_tags:
        raise ValueError("No valid scalar tags to plot.")

    # Plot each tag's values over steps
    plt.figure()
    for t in plot_tags:
        events = ea.Scalars(t)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, label=t)

    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f"TensorBoard Scalars")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "plots", "tensorBoard_scalars.svg"))
    plt.close()

# Create a performance report with all the metrics
def create_performance_report(chpt_dir: str):
    # Get metrics from the lightning logs
    import json
    import glob
    
    # Find the most recent version
    version_dirs = glob.glob(f"{chpt_dir}/lightning_logs/version_*")
    if not version_dirs:
        print("No training logs found.")
        return
    
    latest_version = max(version_dirs, key=lambda x: int(x.split('_')[-1]))
    metrics_path = f"{latest_version}/metrics.csv"
    
    if not os.path.exists(metrics_path):
        print(f"No metrics file found at {metrics_path}")
        return
    
    # Load metrics
    metrics_df = pd.read_csv(metrics_path)
    
    # Filter for test metrics only
    test_metrics = metrics_df[metrics_df['step'].isna()]
    
    # Create a report
    with open(f"{chpt_dir}/test_performance_report.md", 'w') as f:
        f.write("# Model Performance Report\n\n")
        f.write("## Overall Metrics\n\n")
        
        # Add overall metrics
        if 'test_avg_mse' in test_metrics.columns:
            f.write(f"* **Average MSE:** {test_metrics['test_avg_mse'].iloc[-1]:.6f}\n")
        if 'test_avg_rmse' in test_metrics.columns:
            f.write(f"* **Average RMSE:** {test_metrics['test_avg_rmse'].iloc[-1]:.6f}\n")
        if 'test_avg_mae' in test_metrics.columns:
            f.write(f"* **Average MAE:** {test_metrics['test_avg_mae'].iloc[-1]:.6f}\n")
        if 'test_avg_mape' in test_metrics.columns:
            f.write(f"* **Average MAPE:** {test_metrics['test_avg_mape'].iloc[-1]:.6f}%\n")
        if 'test_avg_r2' in test_metrics.columns:
            f.write(f"* **Average R²:** {test_metrics['test_avg_r2'].iloc[-1]:.6f}\n")
        
        f.write("\n## Feature-Specific Metrics\n\n")
        
        # Add feature-specific metrics
        for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
            f.write(f"### {feature}\n\n")
            
            if f'test_{feature}_mse' in test_metrics.columns:
                f.write(f"* **MSE:** {test_metrics[f'test_{feature}_mse'].iloc[-1]:.6f}\n")
            if f'test_{feature}_rmse' in test_metrics.columns:
                f.write(f"* **RMSE:** {test_metrics[f'test_{feature}_rmse'].iloc[-1]:.6f}\n")
            if f'test_{feature}_mae' in test_metrics.columns:
                f.write(f"* **MAE:** {test_metrics[f'test_{feature}_mae'].iloc[-1]:.6f}\n")
            if f'test_{feature}_mape' in test_metrics.columns:
                f.write(f"* **MAPE:** {test_metrics[f'test_{feature}_mape'].iloc[-1]:.6f}%\n")
            if f'test_{feature}_r2' in test_metrics.columns:
                f.write(f"* **R²:** {test_metrics[f'test_{feature}_r2'].iloc[-1]:.6f}\n")
            
            f.write("\n")
        
        # Add prediction plots
        f.write("## Prediction Visualizations\n\n")
        for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
            f.write(f"### {feature} Predictions\n\n")
            f.write(f"![{feature} Predictions](plots/{feature}_predictions.svg)\n\n")
        
        f.write("### All Features (Sample 1)\n\n")
        f.write("![All Features](plots/all_features_predictions.svg)\n\n")


def datasets_from_data_path(data_path):

    streaming_dataset = StreamingDataset(data_path)
    def custom_collate(batch):
        # Filter out None values
        batch = [item for item in batch if item is not None]
        
        if not batch:
            # Return empty tensors if the batch is empty
            return {
                "index": tensor([], dtype=long),
                "inputs": tensor([], dtype=float32),
                "mask": tensor([], dtype=bool),
                "stats": {}
            }
        
        # Process each key separately
        indices = tensor([item["index"] for item in batch], dtype=long)
        
        # Make sure arrays are writable by copying and convert to tensor
        inputs = stack([tensor(np.nan_to_num(item["inputs"].copy(), nan=0.0), dtype=float32) for item in batch])
        masks = stack([tensor(item["mask"].copy(), dtype=bool) for item in batch])
        
        # Get stats (use first non-empty item's stats)
        stats = next((item["stats"] for item in batch if "stats" in item), {})
        
        return {
            "index": indices,
            "inputs": inputs,
            "mask": masks,
            "stats": stats
        }

    logging.info("Total dataset size: %d", len(streaming_dataset)) 

    train_dataset, val_dataset, test_dataset = train_test_split(streaming_dataset, splits=[0.8, 0.1, 0.1])

    logging.info("Train dataset size: %d", len(train_dataset))
    train_dataloader = StreamingDataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True, collate_fn=custom_collate)  # Create DataLoader for training

    logging.info("Validation size: %d", len(val_dataset))
    val_dataloader = StreamingDataLoader(val_dataset, num_workers=4, batch_size=32, shuffle=False, collate_fn=custom_collate)  # Create DataLoader for validation

    test_dataloader = StreamingDataLoader(test_dataset, num_workers=4, batch_size=32, shuffle=False, collate_fn=custom_collate)
    logging.info("Test size: %d", len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader


class TransformerLayer(Module):
    def __init__(self, layer_idx, embed_dim, num_heads, mlp_ratio=4.0, proj_groups=1,
                 dropout=0.05, fast_attention=True):
        super().__init__()
        self.attn = MHA(embed_dim, num_heads, causal=True, layer_idx=layer_idx,
                        num_heads_kv=num_heads//proj_groups,
                        rotary_emb_dim=embed_dim//num_heads,
                        use_flash_attn=fast_attention,
                        return_residual=False, dropout=dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.mlp   = FusedMLP(embed_dim, int(embed_dim*mlp_ratio))
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    


class AutoregressiveTransformerModel(pl.LightningModule):
    def __init__(
            self, 
            input_dim, embed_dim, num_heads, num_layers,
            mlp_ratio=4.0, lr=5e-5, weight_decay=1e-2,
            checkpoint_dir=None, *args, **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()

        # Linear projection layer for input embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(i, embed_dim, num_heads, mlp_ratio)
            for i in range(num_layers)
        ])

        # Final linear layer for output projection
        self.fc_out = nn.Linear(embed_dim, input_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

        # Provide a small weight for the volume feature
        fw = tensor([1.,1.,1.,1.,0.25])
        self.register_buffer('feature_weights', fw)

        # For tracking metrics
        self.train_loss = 0.0
        self.val_loss = 0.0
        
        # Learning rate
        self.lr = lr
        self.weight_decay = weight_decay
        
        # For storing test predictions
        self.test_predictions = []
        self.test_targets = []
        self.test_masks = []
        self.feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.ckpt_dir = checkpoint_dir


    def forward(self, x, mask=None):
        x = self.input_proj(nan_to_num(x, nan=0.0))
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.fc_out(x)

    
    def _calculate_autoregressive_loss(self, preds, targets, mask):
        # shift for next-step prediction
        p, t = preds[:, :-1], targets[:, 1:]
        m = mask[:, 1:].unsqueeze(-1)
        # mse = F.mse_loss(p, t, reduction='none')
        mse = F.smooth_l1_loss(p, t, reduction='none', beta=0.01)
        # apply per-feature weights
        mse = mse * self.feature_weights.view(1,1,-1)
        # mask and reduce
        mse = mse * m
        denom = m.sum().clamp_min(1.0)
        return mse.sum() / denom

    
    def training_step(self, batch, batch_idx):
        """
        Training step for autoregressive prediction
        """
        # Get inputs and mask from batch
        inputs = batch['inputs']
        mask = batch['mask']
        
        # Safety check for NaN in inputs
        if isnan(inputs).any():
            inputs = nan_to_num(inputs, nan=0.0)
        
        # Forward pass to get predictions
        predictions = self(inputs, mask)
        
        # Calculate autoregressive loss
        loss = self._calculate_autoregressive_loss(predictions, inputs, mask)        
        
        # Log the loss for monitoring (don't try to compute gradient norm here)
        self.train_loss = loss
        self.log('train_loss', loss, prog_bar=True)  
        
        return loss
    
    def on_after_backward(self):
        """
        Called after .backward() and before optimizers do anything.
        This is the right place to check gradients.
        """
        # Safely compute gradient norm - after backward pass when gradients exist
        if any(p.grad is not None for p in self.parameters()):
            grad_list = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
            if grad_list:  # Make sure list is not empty
                grad_norm = stack(grad_list).norm(2)
                self.log('grad_norm', grad_norm, prog_bar=True)
            else:
                self.log('grad_norm', tensor(0.0), prog_bar=True)
        else:
            self.log('grad_norm', tensor(0.0), prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for autoregressive prediction
        """
        # Get inputs and mask from batch
        inputs = batch['inputs']
        mask = batch['mask']
        
        # Safety check for NaN in inputs
        if isnan(inputs).any():
            inputs = nan_to_num(inputs, nan=0.0)
        
        # Forward pass to get predictions
        predictions = self(inputs, mask)
        
        # Calculate autoregressive loss
        loss = self._calculate_autoregressive_loss(predictions, inputs, mask)
                
        # Log the loss for monitoring
        self.val_loss = loss
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step for evaluating the model after training
        """
        # Get inputs and mask from batch
        inputs = batch['inputs']
        mask = batch['mask']
        
        # Safety check for NaN in inputs
        if isnan(inputs).any():
            inputs = nan_to_num(inputs, nan=0.0)
        
        # Forward pass to get predictions
        predictions = self(inputs, mask)
        
        # Calculate autoregressive loss
        loss = self._calculate_autoregressive_loss(predictions, inputs, mask)
        
        # Store predictions and targets for later analysis
        # Use detach() and cpu() to avoid memory leaks
        self.test_predictions.append(predictions.detach().cpu())
        self.test_targets.append(inputs.detach().cpu())
        self.test_masks.append(mask.detach().cpu())
        
        # Log the test loss
        self.log('test_loss', loss, prog_bar=True)
        
        return loss
    
    
    # Configuring the optimizer
    def configure_optimizers(self):
        opt = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=0.5,
            patience=2
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def on_test_epoch_end(self):
        """
        Calculate and log test metrics at the end of the test epoch
        """
        # Concatenate all batches
        all_preds = cat(self.test_predictions, dim=0)
        all_targets = cat(self.test_targets, dim=0)
        all_masks = cat(self.test_masks, dim=0)
        
        # For autoregressive prediction, shift by one step
        pred_shifted = all_preds[:, :-1, :]
        target_shifted = all_targets[:, 1:, :]
        mask_shifted = all_masks[:, 1:]
        
        # Convert to numpy for sklearn metrics
        pred_np   = pred_shifted.detach().cpu().to(float32).numpy()
        target_np = target_shifted.detach().cpu().to(float32).numpy()
        mask_np   = mask_shifted.detach().cpu().to(float32).numpy()
        
        # Calculate metrics for each feature
        metrics = {}
        for i, feature_name in enumerate(self.feature_names):
            # Extract predictions and targets for this feature
            feature_preds = pred_np[:, :, i]
            feature_targets = target_np[:, :, i]
            
            # Apply mask to consider only valid positions
            valid_preds = []
            valid_targets = []
            
            # Flatten and filter by mask
            for batch_idx in range(feature_preds.shape[0]):
                for seq_idx in range(feature_preds.shape[1]):
                    if mask_np[batch_idx, seq_idx]:
                        valid_preds.append(feature_preds[batch_idx, seq_idx])
                        valid_targets.append(feature_targets[batch_idx, seq_idx])
            
            # Convert to numpy arrays
            valid_preds = np.array(valid_preds)
            valid_targets = np.array(valid_targets)
            
            if len(valid_preds) > 0:
                # Calculate metrics
                mse = mean_squared_error(valid_targets, valid_preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(valid_targets, valid_preds)
                
                # Calculate MAPE (Mean Absolute Percentage Error) with handling for zeros
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((valid_targets - valid_preds) / np.maximum(np.abs(valid_targets), 1e-8))) * 100
                    mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
                
                # R-squared
                r2 = r2_score(valid_targets, valid_preds)
                
                # Store metrics
                metrics[f"{feature_name}_mse"] = mse
                metrics[f"{feature_name}_rmse"] = rmse
                metrics[f"{feature_name}_mae"] = mae
                metrics[f"{feature_name}_mape"] = mape
                metrics[f"{feature_name}_r2"] = r2
                
                # Log each metric
                self.log(f"test_{feature_name}_mse", mse)
                self.log(f"test_{feature_name}_rmse", rmse)
                self.log(f"test_{feature_name}_mae", mae)
                self.log(f"test_{feature_name}_mape", mape)
                self.log(f"test_{feature_name}_r2", r2)
        
        # Calculate average metrics across all features
        avg_mse = np.mean([metrics[f"{feature}_mse"] for feature in self.feature_names if f"{feature}_mse" in metrics])
        avg_rmse = np.mean([metrics[f"{feature}_rmse"] for feature in self.feature_names if f"{feature}_rmse" in metrics])
        avg_mae = np.mean([metrics[f"{feature}_mae"] for feature in self.feature_names if f"{feature}_mae" in metrics])
        avg_mape = np.mean([metrics[f"{feature}_mape"] for feature in self.feature_names if f"{feature}_mape" in metrics])
        avg_r2 = np.mean([metrics[f"{feature}_r2"] for feature in self.feature_names if f"{feature}_r2" in metrics])
        
        # Log average metrics
        self.log("test_avg_mse", avg_mse)
        self.log("test_avg_rmse", avg_rmse)
        self.log("test_avg_mae", avg_mae)
        self.log("test_avg_mape", avg_mape)
        self.log("test_avg_r2", avg_r2)
        
        # Print summary of test metrics
        print("\n===== TEST METRICS =====")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        print(f"Average MAPE: {avg_mape:.6f}%")
        print(f"Average R²: {avg_r2:.6f}")
        print("=======================\n")
        
        # Create and save visualizations
        self._create_prediction_visualizations(pred_shifted, target_shifted, mask_shifted)
        
        # Clear stored predictions to free memory
        self.test_predictions = []
        self.test_targets = []
        self.test_masks = []

    def _create_prediction_visualizations(self, predictions, targets, masks, num_samples=3):
        """
        Create and save visualization of predictions vs targets
        
        Args:
            predictions: Model predictions [batch_size, seq_len, feature_dim]
            targets: Target values [batch_size, seq_len, feature_dim]
            masks: Boolean masks [batch_size, seq_len]
            num_samples: Number of samples to visualize
        """
        # Convert to numpy for plotting
        preds_np = predictions.detach().cpu().to(float32).numpy()
        targets_np = targets.detach().cpu().to(float32).numpy()
        masks_np = masks.detach().cpu().to(float32).numpy()
        
        # Create directory for plots if it doesn't exist
        import os
        os.makedirs(os.path.join(self.ckpt_dir, "plots"), exist_ok=True)
        
        # Plot for each feature
        for feature_idx, feature_name in enumerate(self.feature_names):
            plt.figure(figsize=(15, 10))
            
            # Plot for a few random samples
            for sample_idx in range(min(num_samples, preds_np.shape[0])):
                # Get predictions and targets for this sample and feature
                sample_preds = preds_np[sample_idx, :, feature_idx]
                sample_targets = targets_np[sample_idx, :, feature_idx]
                sample_mask = masks_np[sample_idx, :]
                
                # Create time index for x-axis
                time_idx = np.arange(len(sample_preds))
                
                # Plot targets
                plt.subplot(num_samples, 1, sample_idx + 1)
                plt.plot(time_idx, sample_targets, 'b-', label='Actual', alpha=0.7)
                
                # Plot predictions (only where mask is True)
                masked_preds = np.where(sample_mask, sample_preds, np.nan)
                plt.plot(time_idx, masked_preds, 'r-', label='Predicted', alpha=0.7)
                
                # Add title and legend
                plt.title(f"Sample {sample_idx+1}: {feature_name}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Calculate metrics for this sample
                valid_indices = np.where(sample_mask)[0]
                if len(valid_indices) > 0:
                    valid_preds = sample_preds[valid_indices]
                    valid_targets = sample_targets[valid_indices]
                    
                    mse = mean_squared_error(valid_targets, valid_preds)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(valid_targets, valid_preds)
                    
                    plt.figtext(0.01, 0.5 - 0.15 * sample_idx, 
                                f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}", 
                                fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.ckpt_dir, "plots", f"{feature_name}_predictions.svg"))
            plt.close()
        
        # Create a summary plot with all features for the first sample
        plt.figure(figsize=(15, 12))
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            plt.subplot(len(self.feature_names), 1, feature_idx + 1)
            
            # Get predictions and targets for first sample and this feature
            sample_preds = preds_np[0, :, feature_idx]
            sample_targets = targets_np[0, :, feature_idx]
            sample_mask = masks_np[0, :]
            
            # Create time index for x-axis
            time_idx = np.arange(len(sample_preds))
            
            # Plot targets
            plt.plot(time_idx, sample_targets, 'b-', label='Actual', alpha=0.7)
            
            # Plot predictions (only where mask is True)
            masked_preds = np.where(sample_mask, sample_preds, np.nan)
            plt.plot(time_idx, masked_preds, 'r-', label='Predicted', alpha=0.7)
            
            # Add title and legend
            plt.title(f"{feature_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.ckpt_dir, "plots", "all_features_predictions.svg"))
        plt.close()



def main(config_path):
    """
    Main function to run the training and testing of the autoregressive transformer model.
    
    Args:
        config_path (str): Path to the YAML configuration file. Will be used to load parameters for the model and training.

    Note:
        The configuration file should contain the following keys:
            - input_dim: the number of features in the input data
            - embed_dim: the dimension of the embedding space
            - num_heads: the number of attention heads
            - num_layers: the number of transformer layers
            - lr: learning rate for the optimizer
            - weight_decay: weight decay for the optimizer
            - limit_val_batches: the number of batches to limit validation
            - accumulate_grad_batches: number of batches to accumulate gradients
            - gradient_clip_val: value for gradient clipping
            - batch_size: batch size for training 
            - sequence_length: length of the input sequences
            - num_workers: number of workers for data loading
            - chkp_dir: directory to save checkpoints
            - nodes: number of nodes for distributed training
            - devices: number of devices for distributed training
            - accelerator: accelerator type (e.g., 'gpu', 'cpu')
            - strategy: strategy for distributed training.
            - precision: precision for training (e.g., 16 for half precision)
            - epochs: number of training epochs
            - log_every_n_steps: log every n steps
            - data_dir: directory for the dataset
            - data_splits: splits for training and validation data
            - seed: random seed for reproducibility
    """

    # Load configuration parameters
    with open(config_path, 'r') as file:
        parameters = yaml.safe_load(file)

    # Create checkpoint directory if it doesn't exist
    make_directory(parameters['chkp_dir'])

    # Set random seed for reproducibility
    pl.seed_everything(parameters['seed'])

    # Load datasets
    train_dataloader, val_dataloader, test_dataloader = datasets_from_data_path(parameters['data_dir'])
    # Build model
    model = AutoregressiveTransformerModel(**parameters)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=parameters['chkp_dir'],
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(
        limit_val_batches=parameters['limit_val_batches'],
        max_epochs=parameters['epochs'], 
        accumulate_grad_batches=parameters['accumulate_grad_batches'], 
        gradient_clip_val=parameters['gradient_clip_val'],  
        default_root_dir=parameters['chkp_dir'],
        precision=parameters['precision'],
        log_every_n_steps=parameters['log_every_n_steps'],
        accelerator=parameters['accelerator'],
        devices=parameters['devices'],
        strategy=parameters['strategy'],
        num_nodes=parameters['nodes'],
        callbacks=[early_stop_callback, checkpoint_callback]
    ) 

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    

    # Test the model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path}")
        model = AutoregressiveTransformerModel.load_from_checkpoint(best_model_path)

    trainer.test(model, test_dataloader)

    # Create performance report
    create_performance_report(parameters['chkp_dir'])
    # Plot TensorBoard scalars
    plot_tensorboard_scalars(parameters['chkp_dir'])


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train and test an autoregressive transformer model.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/transformer_config.yaml", 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)