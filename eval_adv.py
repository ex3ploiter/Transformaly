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
from tqdm import tqdm
from simba import *
import faiss
from pytorch_pretrained_vit.model import AnomalyViT, ViT


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


def get_features(model, data_loader, early_break=-1):
    pretrained_features = []
    for i, (data, _) in enumerate(tqdm(data_loader)):
        if early_break > 0 and early_break < i:
            break

        encoded_outputs = model(data.to('cuda'))
        pretrained_features.append(encoded_outputs.detach().cpu().numpy())

    pretrained_features = np.concatenate(pretrained_features)
    return pretrained_features

def load_model(model_path,VIT_MODEL_NAME):
    # model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
    # model.fc = Identity()
    # model.to('cuda')

    # model_checkpoint_path = join(model_path, 'last_full_finetuned_model_state_dict.pkl')
    # if os.path.exists(model_checkpoint_path):
    #     model_state_dict = torch.load(model_checkpoint_path)
    #     model.load_state_dict(model_state_dict)

    model = ViT('B_16_imagenet1k', pretrained=True)
    model.fc = Identity()
    model.eval()    
    model.cuda()


    return model


def get_score_adv( test_loader,train_features,model):
    model.cuda()
    x_model=Wrap_Model_1(model=model,train_features=train_features)    
    
    t=[]
    l = []
    image_size = 384
    attack = SimBA(x_model, 'imagenet', image_size)
    
    for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader)):
        imgs = imgs.to('cuda')
        labels = labels.to('cuda')
        
        imgs, _, _, _, _, _ = attack.simba_batch(
                imgs, labels, 1, 384, 7, 2/255, linf_bound=0,
                order='rand', targeted=False, pixel_attack=True, log_every=0)
        t.append(x_model(imgs))
        l.append(labels)

    t = np.concatenate(t)
    l = torch.cat(l).cpu().detach().numpy()
        
    auc=roc_auc_score(l, t)
    
    print("ADV AUC: ",auc)

    return auc


def main():
    args=arg_prse()    
    if args['dataset'] == 'BrainMRI':
        _class=2 

    elif  args['dataset'] == 'X-ray':
         _class=0  

    elif  args['dataset'] == 'Head-CT':
        _class=1     
    args['_class'] = _class

    BASE_PATH = 'experiments'
    base_feature_path = join(
        BASE_PATH,
        f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}/class_{_class}')    
    model_path = join(base_feature_path, 'model')
    VIT_MODEL_NAME = 'B_16_imagenet1k'
    model=load_model(model_path,VIT_MODEL_NAME)

    
    with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                    f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                    'train_pretrained_ViT_features.npy'), 'rb') as f:
        train_features = np.load(f)    
    
    trainset, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                data_path=args['data_path'],
                                                one_vs_rest=args['unimodal'],
                                                _class=args['_class'],
                                                normal_test_sample_only=False,
                                                use_imagenet=args['use_imagenet'],
                                                )

    # print(testset[0][0].shape)

    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=args['batch_size'],
                                                shuffle=False)    
    

    
    
    # x_model=Wrap_Model_1(model=model)    
    

    get_score_adv(test_loader=test_loader,train_features=train_features,model=model)





class Wrap_Model_1(torch.nn.Module):
    def __init__(self, model,train_features):
        super().__init__()

        self.train_features=train_features
        
        
        self.model=model
        self.model = self.model.eval()

    def knn_score(self,train_set, test_set, n_neighbours=2):
        """
        Calculates the KNN distance
        """
        index = faiss.IndexFlatL2(train_set.shape[1])
        index.add(train_set)
        dist, _ = index.search(test_set, n_neighbours)
        return np.sum(dist, axis=1)

    def get_features(self,model, data):        
    
        encoded_outputs = model(data.to('cuda'))
        return encoded_outputs.detach().cpu().numpy()
    
    def forward(self, input):
        test_features=self.get_features(self.model,input)
        distances = self.knn_score(self.train_features, test_features, n_neighbours=2)

        return distances        
        

class Wrap_Model_2(torch.nn.Module):
    def __init__(self, model,gmm=None):
        super().__init__()

        # self.train_finetuned_features=train_finetuned_features
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
    
    
main()    