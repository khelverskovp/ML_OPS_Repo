# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    Xtrain = np.load(input_filepath + "/train_0.npz")["images"]
    Ytrain = np.load(input_filepath + "/train_0.npz")["labels"]

    for i in range(1,5):
        Xnext = np.load(input_filepath + "/train_{}.npz".format(i))["images"]
        Xtrain = np.concatenate((Xtrain,Xnext),axis=0)
        Ynext = np.load(input_filepath + "/train_{}.npz".format(i))["labels"]
        Ytrain = np.concatenate((Ytrain,Ynext),axis=0)

    Xtest = np.load(input_filepath + "/test.npz")["images"]
    Ytest = np.load(input_filepath + "/test.npz")["labels"]

    #convert to tensors
    Xtrain = torch.from_numpy(Xtrain).float()
    Ytrain = torch.from_numpy(Ytrain).float()
    Xtest = torch.from_numpy(Xtest).float()
    Ytest = torch.from_numpy(Ytest).float()

    train= torch.utils.data.TensorDataset(Xtrain, Ytrain)
    test = torch.utils.data.TensorDataset(Xtest,Ytest)

    torch.save(train, output_filepath)
    torch.save(test, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
