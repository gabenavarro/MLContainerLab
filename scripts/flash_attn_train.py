import argparse
import logging
import os
import numpy as np
from litdata import StreamingDataset, StreamingDataLoader, train_test_split
from torch import tensor, float32, long, bool, stack


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory {path} created.")


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


