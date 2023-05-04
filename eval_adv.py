"""
Transformaly Evaluation Script
"""
import os
import argparse
import logging
from os.path import join
import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import mixture
import torch.nn
from utils import print_and_add_to_log, get_datasets_for_ViT, \
    Identity, get_finetuned_features
from pytorch_pretrained_vit.model import AnomalyViT
from torch import nn

def arg_prse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--data_path', default='./data/', help='Path to the dataset')
    parser.add_argument('--whitening_threshold', default=0.9, type=float,
                        help='Explained variance of the whitening process')
    parser.add_argument('--unimodal', default=False, action='store_true',
                        help='Use the unimodal settings')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser_args = parser.parse_args()
    args = vars(parser_args)

    args['use_layer_outputs'] = list(range(2, 12))
    args['use_imagenet'] = True
    BASE_PATH = 'experiments'

    return   args  

def load_model(model_path,VIT_MODEL_NAME):
    model_checkpoint_path = join(model_path, 'best_full_finetuned_model_state_dict.pkl')
    model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
    model.fc = Identity()
    model_state_dict = torch.load(model_checkpoint_path)
    ret = model.load_state_dict(model_state_dict)
    print_and_add_to_log(
        'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys),
        logging)
    print_and_add_to_log(
        'Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys),
        logging)
    print_and_add_to_log("model loadded from checkpoint here:", logging)
    print_and_add_to_log(model_checkpoint_path, logging)
    model = model.to('cuda')
    model.eval()
    return model

def main():
    args=arg_prse()
    BASE_PATH = 'experiments'
    base_feature_path = join(
        BASE_PATH,
        f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}/class_{_class}')    
    model_path = join(base_feature_path, 'model')
    model=load_model()
    

class Wrap_Model(torch.nn.Module):
    def __init__(self, train_finetuned_features, test_finetuned_features,model,gmm=None):
        super().__init__()

        self.train_finetuned_features=train_finetuned_features
        # self.test_finetuned_features=test_finetuned_features
        self.gmm=gmm
        self.model=model
        self.model = self.model.eval()

    def get_finetuned_features(self,inputs,seed = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # model = model.to('cuda')
        criterion = nn.MSELoss(reduce=False)

        # start eval
        # model = model.eval()
        

        with torch.no_grad():
            outputs_recon_scores = []
            inputs = inputs.to('cuda')

            origin_block_outputs, cloned_block_outputs = self.model(inputs)
            loss = criterion(cloned_block_outputs, origin_block_outputs)
            loss = torch.mean(loss, [2, 3])
            loss = loss.permute(1, 0)
            outputs_recon_scores.extend(-1 * loss.detach().cpu().data.numpy())

            

            del inputs, origin_block_outputs, cloned_block_outputs, loss
            torch.cuda.empty_cache()
            

        return np.array(outputs_recon_scores)
    def forward(self, input):
        test_finetuned_features=self.get_finetuned_features(input)
        test_finetuned_samples_likelihood=self.gmm.score_samples(test_finetuned_features)

        return test_finetuned_samples_likelihood        