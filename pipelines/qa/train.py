import os
import logging
import shutil
import pathlib

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# try:
from joint_bert.main import main, parse_args
# except:
#     logger.info("joint bert module is not available yet")

if __name__ == '__main__':
    '''
    python train.py --task naive \
                  --model_type bert \
                  --model_dir outputs/model \
                  --data_dir processed \
                  --output_data_dir outputs/data \
                  --num_train_epochs 1 \
                  --save_steps 5
    '''
    args = parse_args()
    args.do_train = 'True'
    args.do_eval = 'False'
    # Convert relative path to absolute path before pass on
    args.data_dir = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    args.output_data_dir = os.path.abspath(args.output_data_dir)
    
    print(f"Files under data dir {args.data_dir}: {os.listdir(args.data_dir)}")
    if len(os.listdir(args.data_dir))==1 and os.listdir(args.data_dir)[0].endswith('.tar.gz'):
        fn = os.path.join(args.data_dir, os.listdir(args.data_dir)[0])
        print(f"Unzipping {fn} into {args.data_dir}:")
        os.system(f"tar -zxvf {fn} -C {args.data_dir}")
        os.system(f"mv {os.path.join(args.data_dir, 'processed')}/* {args.data_dir}/")
        os.system(f"rmdir {os.path.join(args.data_dir, 'processed')}")
        os.system(f"rm {fn}")
    print(f"Files under data dir {args.data_dir}: {os.listdir(args.data_dir)}")
    main(args)

    # Push inference code to model dir
    shutil.copytree('./joint_bert/', f"{args.model_dir}/code/joint_bert/") # copytree will create folder code, but raise an error if it exists
    shutil.copy('./inference.py', f"{args.model_dir}/code/")
    shutil.copy('./evaluate.py', f"{args.model_dir}/code/")
    shutil.copy('./requirements.txt', f"{args.model_dir}/code/")
    # copy train data to model dir, 
    pathlib.Path(f"{args.model_dir}/train_meta").mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{args.data_dir}/intent_label.txt", f"{args.model_dir}/train_meta/")
    shutil.copy(f"{args.data_dir}/slot_label.txt", f"{args.model_dir}/train_meta/")
    shutil.copytree(f"{args.data_dir}/schema/", f"{args.model_dir}/schema/")
