
"""An workflow pipeline script for training question understanding model.

                                                   . RegisterModel
                                               . -
                                              .    . CreateModel
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

import os
import time
from datetime import datetime

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import (
    TrainingInput,
    CreateModelInput,
    TransformInput
)

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.steps import CacheConfig
from sagemaker.debugger import Rule, ProfilerRule, rule_configs
from sagemaker.debugger import DebuggerHookConfig
from sagemaker.debugger import ProfilerConfig, FrameworkProfile
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch import PyTorchModel
from sagemaker.model import FrameworkModel


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print(f'BASE_DIR: {BASE_DIR}')

def get_sagemaker_client(region):
    """
     Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

def get_step_processing(bucket, region, role, params):
    '''
    params: 
        input_data
        output_dir
        validation_split
        test_split
        processing_instance_count
        processing_instance_type
    '''
    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=params['processing_instance_type'],
        instance_count=params['processing_instance_count'],
        env={"AWS_DEFAULT_REGION": region},
    )
    processing_inputs = [
        ProcessingInput(
            input_name="raw",
            source=params['input_data'],
            destination="/opt/ml/processing/qa/data/raw",
            s3_data_distribution_type="ShardedByS3Key",
        )
    ]
    processing_outputs = [
        ProcessingOutput(
            output_name="train",
            destination = params['output_dir'],
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/qa/data/processed", # processed data in preprocessing should be saved to this folder
        )
    ]
    processing_step = ProcessingStep(
        name="Processing",
        code=os.path.join(BASE_DIR, "preprocess.py"),
        processor=processor,
        inputs=processing_inputs,
        outputs=processing_outputs,
        job_arguments=[
            "--input-data",
            processing_inputs[0].destination, # /opt/ml/processing/qa/data/raw
            "--validation-split",
            params['validation_split'],
            "--test-split",
            params['test_split']
        ],
    )
    return processing_step


def get_step_training(bucket, region, role, params, dependencies):
    '''
    params:
        train_instance_type
        train_instance_count
        epochs
        learning_rate
        batch_size
        max_seq_len
    dependencies: 
        'step_process': processing_step
    '''
    train_instance_type = params['train_instance_type']
    train_instance_count = params['train_instance_count']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    max_seq_len = params['max_seq_len']

    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path='s3://sm-nlp-data/nlu/outputs/tb',
        container_local_output_path='/root/ckgqa-p-kiqtyrraeiec/sagemaker-ckgqa-p-kiqtyrraeiec-modelbuild/pipelines/question_ansering/output/tb'
    )

    # Filter out metrics from output
    metric_definitions = [
        {'Name': 'eval:intent_acc', 'Regex': 'intent_acc = ([0-9\\.]+)'},
        {'Name': 'eval:loss', 'Regex': 'loss = ([0-9\\.]+)'},
        {'Name': 'eval:semantic_frame_acc', 'Regex': 'sementic_frame_acc = ([0-9\\.]+)'},
        {'Name': 'eval:slot_f1', 'Regex': 'slot_f1 = ([0-9\\.]+)'},
        {'Name': 'eval:slot_precision', 'Regex': 'slot_precision = ([0-9\\.]+)'},
        {'Name': 'eval:slot_recall', 'Regex': 'slot_recall = ([0-9\\.]+)'}
    ]
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path="s3://{}/nlu/debug".format(bucket),
    )
    profiler_config = ProfilerConfig(
        system_monitor_interval_millis=500,
        framework_profile_params=FrameworkProfile(local_path="/opt/ml/output/profiler/", start_step=5, num_steps=10),
    )
    rules = [ProfilerRule.sagemaker(rule_configs.ProfilerReport())]
    estimator = PyTorch(
        entry_point = 'train.py',
        role=role,
        instance_type=train_instance_type, # ml.c5.4xlarge, ml.g4dn.4xlarge
        instance_count=train_instance_count,
        framework_version='1.8.1',
        py_version='py3',
        source_dir=BASE_DIR,
        output_path=f"s3://{bucket}/nlu/outputs",
        code_location=f"s3://{bucket}/nlu/source/train",
        metric_definitions = metric_definitions,
        hyperparameters={
            'task': 'naive',
            'model_type': 'bert',
            'train_batch_size': batch_size,
            'max_seq_len': max_seq_len,
            'learning_rate': learning_rate,
            'num_train_epochs': epochs
        },
        debugger_hook_config=debugger_hook_config,
        profiler_config=profiler_config,
        rules=rules
    )
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    training_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=dependencies['step_process'].properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
        },
        cache_config=cache_config,
    )
    return training_step


def get_step_evaluation(bucket, region, role, params, dependencies):
    '''
    params:
        evaluation_instance_count
        evaluation_instance_type
    dependencies: 
        'step_train'
        'step_process'
    '''
    evaluation_instance_count = params['evaluation_instance_count']
    evaluation_instance_type = params['evaluation_instance_type']
    
    evaluation_processor = SKLearnProcessor(
        role=role,
        framework_version="0.23-1",
        instance_type=evaluation_instance_type,
        instance_count=evaluation_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        max_runtime_in_seconds=7200,
    )
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="metrics", path="evaluation.json")
    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        code=os.path.join(BASE_DIR, "evaluate.py"),
        inputs=[
            ProcessingInput(
                input_name='model',
                source=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
                # source='s3://sm-nlp-data/nlu/outputs/pipelines-mpyz3z8uxzno-Train-rzHTDRQ9rJ/output/model.tar.gz',
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                input_name='data',
                source=dependencies['step_process'].properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                # source='s3://sm-nlp-data/nlu/data/processed/',
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/metrics/"
            ),
        ],
        job_arguments=[
            "--model_dir", "/opt/ml/processing/input/model/",
            "--task", "naive",
            "--model_type", "bert",
            "--output_data_dir", "/opt/ml/processing/output/metrics/",
            "--data_dir", "/opt/ml/processing/input/data"
        ],
        property_files=[evaluation_report],
    )
    return evaluation_step


def get_step_create_model(bucket, region, role, sess, params, dependencies):
    '''
    params:
        inference_instance_type
    dependencies: 'step_train'
    '''
    inference_instance_type = params['inference_instance_type']
    model_name = "qa-model-{}".format(int(time.time()))
    model = PyTorchModel(
        name=model_name,
        model_data=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
        framework_version='1.3.1',
        py_version='py3',
        role=role,
        entry_point='inference.py',
        source_dir=BASE_DIR,
        sagemaker_session=sess
    )
    create_inputs = CreateModelInput(
        instance_type=inference_instance_type,
        accelerator_type="ml.eia1.medium",
    )
    step_create_model = CreateModelStep(
        name="CreateQAModel",
        model=model,
        inputs=create_inputs,
    )
    return step_create_model


def get_step_register_model(model_package_group_name, params, dependencies):
    '''
    params:
        model_approval_status
        deploy_instance_type
    dependencies: 
        'step_evaluate'
        'step_train'
    '''
    model_approval_status = params['model_approval_status']
    deploy_instance_type = params['deploy_instance_type']
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                dependencies['step_evaluate'].arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="QARegisterModel",
        estimator=dependencies['step_train'].estimator,
        model_data=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[deploy_instance_type],
        transform_instances=["ml.c5.4xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        description=f"Created on {datetime.now()}"
    )
    return step_register

def get_step_condition(evaluation_report, params, dependencies):
    '''
    params:
        min_intent_acc
        min_slot_f1
    dependencies: 
        'step_evaluate'
        'step_register'
        'step_create_model'
    '''
    min_intent_acc = params['min_intent_acc']
    min_slot_f1 = params['min_slot_f1']
    min_intent_acc_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=dependencies['step_evaluate'],
            property_file=evaluation_report,
            json_path="intent_acc",
        ),
        right=min_intent_acc,  # accuracy
    )
    min_slot_f1_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=dependencies['step_evaluate'],
            property_file=evaluation_report,
            json_path="slot_f1",
        ),
        right=min_slot_f1,  # accuracy
    )
    condition_step = ConditionStep(
        name="IntentAndSlotCondition",
        conditions=[min_intent_acc_condition, min_slot_f1_condition],
        if_steps=[dependencies['step_register'], dependencies['step_create_model']],  # success, continue with model registration
        else_steps=[],  # fail, end the pipeline
    )
    return condition_step


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket='sm-nlp-data',
    model_package_group_name="QAPackageGroup",
    pipeline_name="QuestionUnderstandingPipeline",
    base_job_prefix="qa",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    pipeline_name = pipeline_name + str(int(time.time()))
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    sess = sagemaker_session

    raw_input_data_s3_uri = "s3://{}/nlu/data/qa_raw.zip".format(default_bucket)
    processed_data_s3_uri = "s3://{}/nlu/data/processed/".format(default_bucket)
    # preprocessing parameters
    input_data = ParameterString(name="InputData", default_value=raw_input_data_s3_uri)
    output_dir = ParameterString(name="ProcessingOutputData", default_value=processed_data_s3_uri)
    validation_split = ParameterString(name="ValidationSplit", default_value='0.1')
    test_split = ParameterString(name="TestSplit", default_value='0.1')
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.c5.2xlarge")

    # training parameters
    train_instance_type = ParameterString(name="TrainInstanceType", default_value="ml.g4dn.4xlarge")
    train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=1)
    epochs = ParameterString(name="Epochs", default_value='10')
    learning_rate = ParameterString(name="LearningRate", default_value='5e-5')
    batch_size = ParameterString(name="BatchSize", default_value='64')
    max_seq_len = ParameterString(name="MaxSeqLength", default_value='50')

    # evaluation parameters
    evaluation_instance_count = ParameterInteger(name="EvaluationInstanceCount", default_value=1)
    evaluation_instance_type = ParameterString(name="EvaluationInstanceType", default_value="ml.c5.2xlarge")

    # create model parameters
    inference_instance_type = ParameterString(name="InferenceInstanceType", default_value="ml.c5.4xlarge")

    # register model parameters
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    deploy_instance_type = ParameterString(name="DeployInstanceType", default_value="ml.m4.xlarge")
    # deploy_instance_count = ParameterInteger(name="DeployInstanceCount", default_value=1)

    # condition parameters
    min_intent_acc = ParameterFloat(name="MinIntentAccuracy", default_value=0.9)
    min_slot_f1 = ParameterFloat(name="MinSlotF1", default_value=0.95)

    step_process = get_step_processing(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'input_data': input_data,
            'output_dir': output_dir,
            'validation_split': validation_split,
            'test_split': test_split,
            'processing_instance_count': processing_instance_count,
            'processing_instance_type': processing_instance_type
        }
    )

    step_train = get_step_training(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'train_instance_type': train_instance_type,
            'train_instance_count': train_instance_count,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_seq_len': max_seq_len
        },
        dependencies={
            'step_process': step_process
        }
    )

    step_evaluate = get_step_evaluation(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'evaluation_instance_count': evaluation_instance_count,
            'evaluation_instance_type': evaluation_instance_type
        },
        dependencies={
            'step_train': step_train,
            'step_process': step_process
        }
    )

    step_register_model = get_step_register_model(
        model_package_group_name=model_package_group_name,
        params={
            'model_approval_status': model_approval_status,
            'deploy_instance_type': deploy_instance_type
        },
        dependencies={
            'step_evaluate': step_evaluate,
            'step_train': step_train
        }
    )

    step_create_model = get_step_create_model(
        bucket=default_bucket,
        region=region,
        role=role,
        sess=sess,
        params={
            'inference_instance_type': inference_instance_type
        },
        dependencies={
            'step_train': step_train
        }
    )

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="metrics", path="evaluation.json")
    step_condition = get_step_condition(
        evaluation_report=evaluation_report,
        params={
            'min_intent_acc': min_intent_acc,
            'min_slot_f1': min_slot_f1
        },
        dependencies={
            'step_evaluate': step_evaluate,
            'step_register': step_register_model,
            'step_create_model': step_create_model,
        }
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            output_dir,
            validation_split,
            test_split,
            processing_instance_count,
            processing_instance_type,
            
            train_instance_type,
            train_instance_count,
            epochs,
            learning_rate,
            batch_size,
            max_seq_len,

            evaluation_instance_count,
            evaluation_instance_type,

            inference_instance_type,

            model_approval_status,
            deploy_instance_type,

            min_intent_acc,
            min_slot_f1
        ],
        steps=[step_process, step_train, step_evaluate, step_condition],
        sagemaker_session=sess,
    )
    return pipeline