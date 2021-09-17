import os
import logging
import shutil

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

try:
    from joint_bert.main import main, parse_args
except:
    logger.info("joint bert module is not available yet")

if __name__ == '__main__':
    '''
   
    '''
    args = parse_args()

    # args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    
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
    shutil.copytree('./model/', f"{args.model_dir}/code/model/") # copytree will create folder code, but raise an error if it exists
    shutil.copytree('./joint_bert/', f"{args.model_dir}/code/joint_bert/")
