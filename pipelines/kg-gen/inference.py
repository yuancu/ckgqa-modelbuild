
import json
import sys
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import SubjectModel, ObjectModel
from dataset import DevDataset, dev_collate_fn
from utils import extract_spoes


bert_model_name = 'bert-base-chinese'
max_sent_len = 128

###################################
### SAGEMKAER LOAD MODEL FUNCTION
###################################

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subject_path = os.path.join(model_dir, 'subject.pt')
    object_path = os.path.join(model_dir, 'object.pt')
    id2predicate_path = os.path.join(model_dir, 'resources', 'id2predicate.json')
    subject_model = torch.load(subject_path, map_location=device)
    object_model = torch.load(object_path, map_location=device)
    id2predicate = json.load(open(id2predicate_path))
    return (subject_model, object_model, id2predicate)


###################################
### SAGEMKAER PREDICT FUNCTION
###################################

def predict_fn(input_data, model):
    '''
    Args:
    input_data (list(dict)): list of text and spoes pairs [{'text':, 'spo_list':},]
    model (tuple): subject model and object model
    
    Returns:
    rel_df (pd.Dataframe): a panda data frame that saves (subject, predicate, object) pairs
    '''
    subject_model, object_model, id2predicate = model
    subject_model.eval()
    object_model.eval()
    
    print("type(input_data): {}".format(type(input_data)))
    
    dataset = DevDataset(data, bert_model_name, 128)
    loader = DataLoader(
        dataset=dev_dataset,  
        batch_size=256, 
        shuffle=False,
        num_workers=1,
        collate_fn=dev_collate_fn,
        multiprocessing_context='spawn',
    )
    
    rel_df = pd.DataFrame({'subject':[], 'predicate':[], 'object':[]})
    for batch in loader:
        texts, tokens, spoes, att_masks, offset_mappings = batch
        rels = extract_spoes(texts, tokens, offset_mappings, subject_model, object_model, \
                              id2predicate, attention_mask=att_masks)
        for rel in rels:
            rel_df.loc[len(rel_df)] = rel

    return rel_df


###################################
### SAGEMKAER MODEL INPUT FUNCTION
###################################

def input_fn(serialized_input_data, content_type="application/jsonlines"):
    return serialized_input_data


###################################
### SAGEMKAER MODEL OUTPUT FUNCTION
###################################

def output_fn(prediction_output, accept="application/jsonlines"):
    return prediction_output, accept

