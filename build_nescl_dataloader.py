import argparse
import os
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders
from recbole.utils import init_logger, init_seed
import logging

def build_dataloader(model, dataset):
    # This automatically invokes customized_dataset.py logic due to Config mappings.
    config = Config(model=model, dataset=dataset, config_file_list=[f"config/{dataset}/{dataset}.yaml"])
    
    # Initialize logger
    init_logger(config)
    logger = logging.getLogger()
    
    # Init seed
    init_seed(config['seed'], config['reproducibility'])

    logger.info(f"Building Dataset for model: {model}, dataset: {dataset}")
    # create_dataset automatically picks up NESCLDataset/SUPCCLDataset
    dataset_obj = create_dataset(config)
    
    logger.info(f"Dataset Built. Info:\n{dataset_obj}")
    
    logger.info("Splitting dataset and preparing dataloaders...")
    train_data, valid_data, test_data = data_preparation(config, dataset_obj, save=False)
    
    # Manually save dataloader using Recbole's util exactly as run_recbole_autodl expects
    save_path = config['checkpoint_dir'] if 'checkpoint_dir' in config and config['checkpoint_dir'] else 'saved'
    os.makedirs(save_path, exist_ok=True)
    
    # Matching exactly the format `dataset-for-model-dataloader.pth`
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    
    logger.info(f"Saving dataloaders to {file_path}")
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump((train_data, valid_data, test_data), f)
        
    logger.info("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SUPCCL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='gowalla', help='name of datasets')
    args = parser.parse_args()
    build_dataloader(args.model, args.dataset)
