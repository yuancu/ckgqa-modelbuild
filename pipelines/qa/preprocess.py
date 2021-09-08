"""Feature engineers the questions."""
import argparse
import logging
import os
import pathlib
import json
import shutil
import subprocess

import boto3
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# helper functions to upload data to s3
def write_to_s3(filename, bucket, prefix):
    import boto3
    # put one file in a separate folder. This is helpful if you read and prepare data with Athena
    filename_key = filename.split("/")[-1]
    key = os.path.join(prefix, filename_key)
    s3 = boto3.resource('s3')
    return s3.Bucket(bucket).upload_file(filename, key)


def upload_to_s3(bucket, prefix, filename):
    url = "s3://{}/{}".format(bucket, os.path.join(prefix, filename.split('/')[-1]))
    print("Writing to {}".format(url))
    write_to_s3(filename, bucket, prefix)
    

def parse_args():
    # Local testing with CLI args
    parser = argparse.ArgumentParser(description="Preprocess")
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=False)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    return parser.parse_args()


def process(raw, processed, val_split, test_split):
    '''
    
    '''

    # Read in files
    # questions:            questions in natural language
    # question_bios:        boundary, inside, outside
    # question_intentions   corresponding intentions to every question
    # question_templates
    # domain_specific_slot_labels
    # domain_specific_intentions
    with open(os.path.join(raw, 'seq.in')) as f:
        questions = f.readlines()
        questions = [l.rstrip() for l in questions]
    with open(os.path.join(raw, 'seq.out')) as f:
        question_bios = f.readlines()
        question_bios = [l.rstrip().split() for l in question_bios]
    with open(os.path.join(raw, 'label')) as f:
        question_intentions = f.readlines()
        question_intentions = [l.rstrip() for l in question_intentions]
    with open(os.path.join(raw, 'schema', 'question_templates.json')) as f:
        question_templates = json.load(f)
    with open(os.path.join(raw, 'schema', 'domain_specific_slot_labels.json')) as f:
        domain_specific_slot_labels = json.load(f)
    with open(os.path.join(raw, 'schema', 'domain_specific_intentions.json')) as f:
        domain_specific_intentions = json.load(f)
    
    # Summarize all labels and intentions for classification task
    all_slot_labels = ['PAD', 'UNK', 'O', 'B_name', 'I_name'] # name refers to human name
    all_intentions = ['UNK']
    for labels in domain_specific_slot_labels.values():
        # this keeps orders (compared with using set)
        labels = labels['subject_label'] + labels['object_label']
        all_slot_labels += [label for label in labels if label not in all_slot_labels]
    for intentions in domain_specific_intentions.values():
        if intentions['ask_object'] not in all_intentions:
            all_intentions.append(intentions['ask_object'])
        if 'ask_subject' in intentions and intentions['ask_subject'] not in all_intentions:
            all_intentions.append(intentions['ask_subject'])
    logger.info("slot labels:", all_slot_labels)
    logger.info("intentions:", all_intentions)

    # Split train, val, test
    num_total = len(questions)
    num_val = int(val_split * num_total)
    num_test = int(test_split * num_total)
    num_train = num_total - num_val - num_test
    logger.info(f"Samples for training: {num_train}, for validation: {num_val}, for test: {num_test}")

    train_questions = questions[:num_train]
    train_bios = question_bios[:num_train]
    train_intentions = question_intentions[:num_train]
    dev_questions = questions[num_train:num_train+num_val]
    dev_bios = question_bios[num_train:num_train+num_val]
    dev_intentions = question_intentions[num_train:num_train+num_val]
    test_questions = questions[num_train+num_val:]
    test_bios = question_bios[num_train+num_val:]
    test_intentions = question_intentions[num_train+num_val:]

    data_dict = {'train': (train_questions, train_bios, train_intentions),
                'dev': (dev_questions, dev_bios, dev_intentions),
                'test': (test_questions, test_bios, test_intentions)}
    
    # write processed data into processed folder
    pathlib.Path(processed).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(processed,'intent_label.txt'), 'w') as f:
        for intention in all_intentions:
            f.write("%s\n" % intention)
    with open(os.path.join(processed, 'slot_label.txt'), 'w') as f:
        for slot_label in all_slot_labels:
            f.write("%s\n" % slot_label)
    for item in ['train', 'dev', 'test']:
        pathlib.Path(os.path.join(processed, item)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(processed, item, 'seq.in'), 'w') as f:
            for question in data_dict[item][0]:
                f.write("%s\n" % question)
        with open(os.path.join(processed, item, 'seq.out'), 'w') as f:
            for bio in data_dict[item][1]:
                f.write("%s\n" % ' '.join(bio))
        with open(os.path.join(processed, item, 'label'), 'w') as f:
            for intent in data_dict[item][2]:
                f.write("%s\n" % intent)
    # dirs_exist_ok=True is valid from 3.8
    shutil.copytree(os.path.join(raw, 'schema')+'/', os.path.join(processed, 'schema')+'/')
    

if __name__ == '__main__':
    '''
    Invoke:
    python preprocess --input-data s3://sm-nlp-data/nlu/data/qa_raw.zip --output-dir s3://sm-nlp-data/nlu/data/processed
    '''
    args = parse_args()
    base_dir = "/opt/ml/processing/qa"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    if input_data.startswith('s3://'):
        bucket = input_data.split("/")[2]
        key = "/".join(input_data.split("/")[3:])
        logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
        fn = f"{base_dir}/data/qa_raw.zip"
        s3 = boto3.resource("s3")
        s3.Bucket(bucket).download_file(key, fn)
    else:
        # it should be a local file
        if os.path.isfile(args.input_data):
            logger.info(f"Raw data locates at {args.input_data}")
            fn = args.input_data
        elif os.path.isdir(args.input_data):
            logger.error(f"{args.input_data} should be the file location.\n trying to load {args.input_data}/qa_raw.zip")
            logger.info(f"files under {args.input_data}: {os.listdir(args.input_data)}")
            fn =  f"{args.input_data}/qa_raw.zip"
        else:
            logger.error(f"{args.input_data} doesn't exist")
            logger.info(f"files in parent folder {args.input_data+'/..'}: {os.listdir(args.input_data+'/..')}")
            fn = args.input_data
    
    logger.info("Unzipping dowloaded data...")
    raw = f"{base_dir}/data/raw"
    pathlib.Path(raw).mkdir(parents=True, exist_ok=True)
    os.system(f"unzip {fn} -d {raw}")
    logger.info(f"Data unzipped to {raw}")
    
    processed = f"{base_dir}/data/processed"
    pathlib.Path(processed).mkdir(parents=True, exist_ok=True)
    
    process(raw, processed, args.validation_split, args.test_split)
    
    if args.output_dir is not None:
        # upload processed data as a package
        subprocess.check_call(['tar', '-zcvf', 'processed.tar.gz', processed])
        bucket = args.output_dir.split("/")[2]
        prefix = "/".join(args.output_dir.split("/")[3:])
        upload_to_s3(bucket, prefix, 'processed.tar.gz')
        subprocess.check_call(['aws', 's3', 'cp', processed, args.output_dir])
    