import os
import sys
import json
import argparse
import pathlib
import logging
import subprocess
import tarfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    '''
    Invoke test:
    python evaluate.py --model_dir compressed_model --task naive --output_data_dir outputs/data --data_dir processed
    '''
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument("--model_dir", default=os.environ['SM_MODEL_DIR'], type=str, help="Path to save, load model")
        parser.add_argument("--data_dir", default=os.environ['SM_CHANNEL_TRAIN'], type=str, help="The input data dir")
        parser.add_argument('--output_data_dir', default=os.environ['SM_OUTPUT_DATA_DIR'], type=str, help="The output data dir")
    except:
        parser.add_argument("--model_dir", required=True, type=str, help="Path to save, load model")
        parser.add_argument("--data_dir", required=True, type=str, help="The input data dir")
        parser.add_argument('--output_data_dir', type=str, help="The output data dir")
    parser.add_argument('--do_train', type=str)
    parser.add_argument('--do_eval', type=str)
    args, other_args = parser.parse_known_args()
    
    # Unzip model file
    logger.info(f"Files under model dir {args.model_dir}: {os.listdir(args.model_dir)}")
    if len(os.listdir(args.model_dir))>0 and 'model.tar.gz' in os.listdir(args.model_dir):
        model_tar_path = "{}/model.tar.gz".format(args.model_dir)
        model_tar = tarfile.open(model_tar_path)
        model_tar.extractall(args.model_dir)
        model_tar.close()
        os.unlink(os.path.join(args.model_dir, 'model.tar.gz'))
        logger.info(f"Model dir {args.model_dir} after extraction: {os.listdir(args.model_dir)}")
    if 'code' in os.listdir(args.model_dir):
        code_dir = os.path.join(args.model_dir, 'code')
        logger.info(f"Files under code dir {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(1, code_dir)
        if 'requirements.txt' in os.listdir(code_dir):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", f"{code_dir}/requirements.txt"])
    else:
        raise Exception('code folder should exists in model tar file')
        
    # Hardcode some args
    args.do_eval = 'True'
    args.do_train = 'False'
    # Convert relative path to absolute path before pass on
    args.data_dir = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    args.output_data_dir = os.path.abspath(args.output_data_dir)
    
    from joint_bert.main import main, create_parser 
    
    parser = create_parser()
    # Redefine args with parser from joint_bert
    other_args.extend([
        '--do_eval',
        args.do_eval,
        '--do_train',
        args.do_train,
        '--data_dir',
        args.data_dir,
        '--model_dir',
        args.model_dir,
        '--output_data_dir',
        args.output_data_dir
    ])
    args = parser.parse_args(other_args)
    
    eval_results = main(args)
    pathlib.Path(args.output_data_dir).mkdir(parents=True, exist_ok=True)
    out_fn = os.path.join(args.output_data_dir, 'evaluation.json')
    logger.info(f"Dumping evaluation results to {out_fn}")
    with open(out_fn, 'w') as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
    if os.path.isfile(out_fn):
        logger.info("Dumping succeed")
    else:
        logger.error(f"Failed to save evaluation results at {out_fn}")
