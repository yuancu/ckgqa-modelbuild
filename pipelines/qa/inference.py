import os
import json
import logging
import hashlib
from argparse import Namespace

import torch

from joint_bert.predict import get_args, load_model, read_input_file, predict_helper, convert_input_file_to_tensor_dataset
from joint_bert.utils import load_tokenizer, get_intent_labels, get_slot_labels

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
logger.info(f"Current file path: {dir_path}")
logger.info(f"files under current file path: {os.listdir(dir_path)}")
logger.info(f"Working directory: {cwd}")
logger.info(f"files under working directory: {os.listdir(cwd)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

###################################
### SAGEMKAER LOAD MODEL FUNCTION
###################################

def model_fn(model_dir):
    logger.info(f"model dir is {model_dir}")
    logger.info(f"files under model dir: {os.listdir(model_dir)}")
    for file in os.listdir(model_dir):
        if os.path.isdir(file):
            logger.info(f"files under {file}: {os.listdir(file)}")
    
    pred_config = Namespace(
        model_dir=model_dir
    )
    train_args_bin = os.path.join(pred_config.model_dir, 'training_args.bin')
    logger.info(f"MD5 of training args: {md5(train_args_bin)}")
    print(f"torch version: {torch.__version__}")
    
    args = get_args(pred_config)
    args.data_dir = os.path.join(model_dir, 'train_meta')
    args.model_dir = model_dir
    logger.info(f"Args: {args}")
    
    model = load_model(pred_config, args, device)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)
    
    if torch.__version__ == '1.5.1':
        import torcheia
        model = model.eval()
        # attach_eia() is introduced in PyTorch Elastic Inference 1.5.1,
        model = torcheia.jit.attach_eia(model, 0)
    return (model, args, intent_label_lst, slot_label_lst)


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
    
    pred_config = Namespace(
        batch_size=64
    ) # actually not used in convert_input_file_to_tensor_dataset
    model, args, intent_label_lst, slot_label_lst = model
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    dataset = convert_input_file_to_tensor_dataset(input_data, pred_config, args, tokenizer, pad_token_label_id)
    slot_preds_list, intent_preds = predict_helper(model, dataset, pred_config, args, device, slot_label_lst, pad_token_label_id)

    logger.info("Prediction Done!")
    
    intent_preds_text = [intent_label_lst[intent_pred] for intent_pred in intent_preds]
    
    predict_result = {
        'text': input_data,
        'intentions': intent_preds_text,
        'slot_labels': slot_preds_list
    }
    
    logger.info(f"Results:\nText: {predict_result['text']}\nSlot labels: {predict_result['slot_labels']}")
    logger.info(f"intention[0] type {type(intent_preds[0])}")
    logger.info(f"slot_labels[0][0] type: {type(slot_preds_list[0][0])}")

    return predict_result


###################################
### SAGEMKAER MODEL INPUT FUNCTION
###################################

def input_fn(serialized_input_data, content_type="application/jsonlines"):
    lines = []
    for line in serialized_input_data:
        line = line.strip()
        # words = line.split()
        words = list(line)
        lines.append(words)
    logger.info(f"input data length: {len(lines)}")
    if len(lines) > 1:
        logger.info(f"first piece of data: {lines[0]}")
    return lines


###################################
### SAGEMKAER MODEL OUTPUT FUNCTION
###################################

def output_fn(prediction_output, accept="application/json"):
    return json.dumps(prediction_output, ensure_ascii=False), accept


# test inference code locally
if __name__ == '__main__':
    import jsonlines
    model = model_fn('outputs/model')
    with open('processed/psuedo/seq.in') as f:
        lines = f.readlines()
        lines = input_fn(lines)
    predict_result = predict_fn(lines, model)
    json_out = output_fn(predict_result, accept="application/json")