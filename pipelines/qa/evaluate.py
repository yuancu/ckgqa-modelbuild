import os
import sys
import json
import logging
import subprocess
import tarfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

try:
    from joint_bert.main import main, parse_args
except:
    logger.info("joint bert module is not available yet")

if __name__ == '__main__':
    '''
    Invoke test:
    python evaluate.py --model_dir outputs/model --task naive --output_data_dir outputs/data --data_dir processed
    '''
    args = parse_args()
    args.do_eval = 'True'
    args.do_train = 'False'
    # Convert relative path to absolute path before pass on
    args.data_dir = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    args.output_data_dir = os.path.abspath(args.output_data_dir)
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
        from joint_bert.main import main, parse_args 

    eval_results = main(args)
    out_fn = os.path.join(args.output_data_dir, 'evaluation.json')
    logger.info(f"Dumping evaluation results to {out_fn}")
    with open(out_fn, 'w') as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
    if os.path.isfile(out_fn):
        logger.info("Dumping succeed")
    else:
        logger.error(f"Failed to save evaluation results at {out_fn}")
