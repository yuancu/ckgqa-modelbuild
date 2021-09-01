
"""An workflow pipeline script for Knowledge Graph generation pipeline.

                                                   . RegisterModel
                                               . -
                                              .    . CreateModel -> BatchTransform
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import time

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
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.model import FrameworkModel
from sagemaker.transformer import Transformer

import pandas as pd

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

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
        processing_instance_count
        processing_instance_type
    '''
    input_data = params['input_data']
    output_dir = params['output_dir']
    processing_instance_count = params['processing_instance_count']
    processing_instance_type = params['processing_instance_type']
    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        )
    processing_inputs = [
        ProcessingInput(
            input_name="raw",
            source=input_data,
            destination="/opt/ml/processing/ie/data/raw",
            s3_data_distribution_type="ShardedByS3Key",
            )
        ]
    processing_outputs = [
        ProcessingOutput(
            output_name="train",
            destination = output_dir,
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/ie/data/processed",
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
            processing_inputs[0].destination, # /opt/ml/processing/ie/data/raw
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
    dependencies: 
        'step_process': processing_step
    '''
    train_instance_type = params['train_instance_type']
    train_instance_count = params['train_instance_count']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    # Setup Metrics To Track Model Performance
    metric_definitions = [
        {'Name': 'eval:f1', 'Regex': 'f1: ([0-9\\.]+)'},
        {'Name': 'eval:prec', 'Regex': 'precision: ([0-9\\.]+)'},
        {'Name': 'eval:recall', 'Regex': 'recall: ([0-9\\./]+)'}
    ]
    # Setup Debugger and Profiler
    # Define Debugger Rules as described here: https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path="s3://{}/ie-baseline/debug".format(bucket),
    )
    profiler_config = ProfilerConfig(
        system_monitor_interval_millis=500,
        framework_profile_params=FrameworkProfile(local_path="/opt/ml/output/profiler/", start_step=5, num_steps=10),
    )
    rules = [ProfilerRule.sagemaker(rule_configs.ProfilerReport())]
    # Define a Training Step to Train a Model
    estimator = PyTorch(entry_point=os.path.join(BASE_DIR, 'train.py'),
        source_dir=BASE_DIR,
        role=role,
        instance_type=train_instance_type, # ml.c5.4xlarge, ml.g4dn.4xlarge
        instance_count=train_instance_count,
        framework_version='1.8.1',
        py_version='py3',
        output_path=f"s3://{bucket}/ie-baseline/outputs",
        code_location=f"s3://{bucket}/ie-baseline/source/train", # where custom code will be uploaded 
        hyperparameters={
            'epochs': epochs,
            'use-cuda': True,
            'batch-size': batch_size,
            'learning-rate': learning_rate
        },
        metric_definitions = metric_definitions,
        debugger_hook_config=debugger_hook_config,
        profiler_config=profiler_config,
        rules=rules
    )
    # Setup Pipeline Step Caching
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

    training_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=dependencies['step_process'].properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="application/json",
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
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                input_name='data',
                source=dependencies['step_process'].properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/data",
            ),
            ProcessingInput(
                input_name='source',
                source=dependencies['step_train'].arguments['HyperParameters']['sagemaker_submit_directory'][1:-1],
                destination="/opt/ml/processing/input/source/train"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/metrics/"
            ),
        ],
        job_arguments=[
            "--max-seq-length",
            "128",
            "--source-dir",
            "/opt/ml/processing/input/source/train"
        ],
        property_files=[evaluation_report],
    )
    return evaluation_step


def get_step_create_model(bucket, region, role, sess, params, dependencies):
    '''
    params:
        transform_model_name
        inference_instance_type
    dependencies: 'step_train'
    '''
    transform_model_name = params['transform_model_name']
    inference_instance_type = params['inference_instance_type']
    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="1.8.1",
        py_version="py3",
        instance_type=inference_instance_type,
        image_scope='inference'
    )
    model = FrameworkModel(
        name=transform_model_name,
        image_uri=inference_image_uri,
        entry_point=os.path.join(BASE_DIR, "inference.py"),
        model_data=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sess,
        role=role,
        source_dir=BASE_DIR
    )
    create_inputs = CreateModelInput(
        instance_type="ml.c5.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    step_create_model = CreateModelStep(
        name="CreateKgGenModel",
        model=model,
        inputs=create_inputs,
    )
    return step_create_model


def get_step_transform(bucket, region, role, params, dependencies):
    '''
    params:
        transform_instance_type
        batch_data
    dependencies: 'step_create_model'
    '''
    transform_instance_type = params['transform_instance_type']
    batch_data = params['batch_data']
    transformer = Transformer(
        model_name=dependencies['step_create_model'].properties.ModelName,
        instance_type=transform_instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/ie-baseline/outputs",
    )
    step_transform = TransformStep(
        name="KgTransform", transformer=transformer, inputs=TransformInput(data=batch_data)
    )
    return step_transform



def get_step_register_model(model_package_group_name, params, dependencies):
    '''
    params:
        model_approval_status
        transform_instance_type
        deploy_instance_type
    dependencies: 
        'step_evaluate'
        'step_train'
    '''
    model_approval_status = params['model_approval_status']
    transform_instance_type = params['transform_instance_type']
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
        name="KgRegisterModel",
        estimator=dependencies['step_train'].estimator,
        model_data=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[deploy_instance_type],
        transform_instances=[transform_instance_type],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    return step_register


def get_step_condition(evaluation_report, params, dependencies):
    '''
    params:
        min_f1_value
    dependencies: 
        'step_evaluate'
        'step_register'
        'step_create_model'
        'step_transform'
    '''
    min_f1_value = params['min_f1_value']
    minimum_f1_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=dependencies['step_evaluate'],
            property_file=evaluation_report,
            json_path="f1",
        ),
        right=min_f1_value,  # accuracy
    )
    minimum_f1_condition_step = ConditionStep(
        name="F1Condition",
        conditions=[minimum_f1_condition],
        if_steps=[dependencies['step_register'], dependencies['step_create_model'], \
            dependencies['step_transform']],  # success, continue with model registration
        else_steps=[],  # fail, end the pipeline
    )
    return minimum_f1_condition_step


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="KgGenPackageGroup",
    pipeline_name="KnowledgeGraphGenerationPipeline",
    base_job_prefix="ie",
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
    bucket = 'sm-nlp-data'
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    sess = get_session(region, bucket)
        
    # processing parameters
    raw_input_data_s3_uri = "s3://{}/ie-baseline/raw/DuIE_2_0.zip".format(bucket)
    processed_data_s3_uri = "s3://{}/ie-baseline/processed/".format(bucket)
    input_data = ParameterString(name="InputData", default_value=raw_input_data_s3_uri)
    output_dir = ParameterString(name="OutputData", default_value=processed_data_s3_uri,)
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.c5.2xlarge")

    # train parameters
    train_instance_type = ParameterString(name="TrainInstanceType", default_value="ml.g4dn.16xlarge")
    train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=1)
    epochs = ParameterString(name="Epochs", default_value='20')
    learning_rate = ParameterString(name="LearningRate", default_value='0.005')
    batch_size = ParameterString(name="BatchSize", default_value='64')

    # evaluate parameters
    evaluation_instance_count = ParameterInteger(name="EvaluationInstanceCount", default_value=1)
    evaluation_instance_type = ParameterString(name="EvaluationInstanceType", default_value="ml.c5.2xlarge")

    # create model parameters
    transform_model_name = ParameterString(name="TransformModelName", default_value="transform-model-{}".format(int(time.time())))
    inference_instance_type = ParameterString(name="InferenceInstanceType", default_value="ml.g4dn.16xlarge")
 
    # batch transform parameters
    transform_instance_type = ParameterString(name="TransformInstanceType", default_value="ml.c5.4xlarge")
    batch_data = ParameterString(name="BatchData", default_value='s3://sm-nlp-data/psudo/psudo.json',)

    # register parameters
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    deploy_instance_type = ParameterString(name="DeployInstanceType", default_value="ml.m4.xlarge")

    # condition parameters
    min_f1_value = ParameterFloat(name="MinF1Value", default_value=0.5)


    step_process = get_step_processing(
        bucket=bucket,
        region=region,
        role=role,
        params={
            'input_data': input_data,
            'output_dir': output_dir,
            'processing_instance_count': processing_instance_count,
            'processing_instance_type': processing_instance_type
        })
    
    step_train = get_step_training(
        bucket=bucket,
        region=region,
        role=role,
        params={
            'train_instance_type': train_instance_type,
            'train_instance_count': train_instance_count,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        dependencies={
            'step_process': step_process
        }
    )

    step_evaluate = get_step_evaluation(
        bucket=bucket,
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
            'transform_instance_type': transform_instance_type,
            'deploy_instance_type': deploy_instance_type
        },
        dependencies={
            'step_evaluate': step_evaluate,
            'step_train': step_train
        }
    )

    step_create_model = get_step_create_model(
        bucket=bucket,
        region=region,
        role=role,
        sess=sess,
        params={
            'transform_model_name': transform_model_name,
            'inference_instance_type': inference_instance_type
        },
        dependencies={
            'step_train': step_train
        }
    )

    step_transform = get_step_transform(
        bucket=bucket,
        region=region,
        role=role,
        params={
            'transform_instance_type': transform_instance_type,
            'batch_data': batch_data
        },
        dependencies={
            'step_create_model': step_create_model
        }
    )

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="metrics", path="evaluation.json")

    step_condition = get_step_condition(
        evaluation_report=evaluation_report,
        params={
            'min_f1_value': min_f1_value
        },
        dependencies={
            'step_evaluate': step_evaluate,
            'step_register': step_register_model,
            'step_create_model': step_create_model,
            'step_transform': step_transform
        }
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            output_dir,
            processing_instance_count,
            processing_instance_type,
            
            train_instance_type,
            train_instance_count,
            epochs,
            learning_rate,
            batch_size,

            evaluation_instance_count,
            evaluation_instance_type,

            transform_model_name,
            inference_instance_type,
            
            transform_instance_type,
            batch_data,

            model_approval_status,
            deploy_instance_type,
            
            min_f1_value
        ],
        steps=[step_process, step_train, step_evaluate, step_condition],
        sagemaker_session=sess,
    )
    return pipeline