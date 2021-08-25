"""Feature engineers the DuIE dataset."""
import argparse
import logging
import os
import pathlib
import json

import boto3
from tqdm import tqdm

from utils import upload_to_s3


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def trans(raw, processed):
    # Read raw schema
    schemas = set()
    with open(f"{raw}/schema.json") as f:
        for l in f:
            a = json.loads(l)
            schemas.add(a['predicate'])
    
    # Create id:predicate dicts
    id2predicate = {i+1:j for i,j in enumerate(schemas)}
    id2predicate[0] = 'UNK'
    predicate2id = {j:i for i,j in id2predicate.items()}
    predicate2id['UNK'] = 0

    # Save processed schema
    with open(f"{processed}/schema.json", 'w', encoding='utf-8') as f:
        json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)

    logger.info("Processing raw train data...")
    train_data = []
    with open(f"{raw}/train.json") as f:
        for l in tqdm(f):
            a = json.loads(l)
            train_data.append(
                {
                    'text': a['text'],
                    'spo_list': [(i['subject'], i['predicate'], i['object']['@value']) for i in a['spo_list']]
                }
            )
    
    logger.info(f"Dumping processed train data into {processed}/train.json")
    with open(f"{processed}/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    logger.info("Processing raw dev data...")
    dev_data = []
    with open(f"{raw}/dev.json") as f:
        for l in tqdm(f):
            a = json.loads(l)
            dev_data.append(
                {
                    'text': a['text'],
                    'spo_list': [(i['subject'], i['predicate'], i['object']['@value']) for i in a['spo_list']]
                }
            )
    logger.info(f"Dumping processed dev data into {processed}/dev.json")
    with open(f"{processed}/dev.json", 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    '''
    $ python preprocess.py --input-data s3://sm-nlp-data/ie-baseline/raw/DuIE_2_0.zip --output-dir s3://sm-nlp-data/ie-baseline/train/
    '''
    
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=False)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing/ie"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/duie.zip"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    
    logger.info("Unzipping dowloaded data...")
    raw = f"{base_dir}/data/raw"
    pathlib.Path(raw).mkdir(parents=True, exist_ok=True)
    os.system(f"unzip -j {fn} -d {raw}")
    logger.info(f"Data unzipped to {raw}")
    
    processed = f"{base_dir}/data/processed"
    pathlib.Path(processed).mkdir(parents=True, exist_ok=True)
    
    # Transform raw data
    trans(raw, processed)
    # Delete downloaded and raw data
    os.unlink(fn)
    os.unlink(f"{raw}/schema.json")
    os.unlink(f"{raw}/train.json")
    os.unlink(f"{raw}/dev.json")
    
    # Upload processed data to s3
    if args.output_dir is not None:
        bucket = args.output_dir.split("/")[2]
        prefix = "/".join(args.output_dir.split("/")[3:])
        upload_to_s3(bucket, prefix, f"{processed}/train.json")
        upload_to_s3(bucket, prefix, f"{processed}/dev.json")
        upload_to_s3(bucket, prefix, f"{processed}/schema.json")