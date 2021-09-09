'''Programatically ingest data into Neptune graph database'''
import os
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformed-data', type=str, help='s3 or local path of batch transformed data')
    return parser.parse_known_args()