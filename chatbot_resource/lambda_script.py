'''
Python Lambda function example for Amazon Neptune

Here are some things to notice about the following Python Amazon Lambda example function:
- It uses the backoff module
- It sets pool_size=1 to keep from creating an unnecessary connection pool.
- It sets message_serializer=serializer.GraphSONSerializersV2d0().

Required env variables:
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_SESSION_TOKEN
neptuneEndpoint
neptunePort
nluEndpoint
USE_IAM (optional)
'''

import os
import sys
import backoff
import math
from random import randint
from gremlin_python import statics
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.driver.protocol import GremlinServerError
from gremlin_python.driver import serializer
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.process.traversal import T
from tornado.websocket import WebSocketClosedError
from tornado import httpclient
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials
from types import SimpleNamespace

import json
import sagemaker
from sagemaker.pytorch.model import PyTorchPredictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer


reconnectable_err_msgs = [
    'ReadOnlyViolationException',
    'Server disconnected',
    'Connection refused'
]

retriable_err_msgs = [
    'ConcurrentModificationException'] + reconnectable_err_msgs

network_errors = [WebSocketClosedError, OSError]

retriable_errors = [GremlinServerError] + network_errors


def parse_questions(questions, nlu_endpoint_name):
    '''
    Args:
        questions (list(str)): A list of natural language questions
        nlu_endpoint_name
    '''
    sess = sagemaker.Session()
    predictor = PyTorchPredictor(
        endpoint_name=nlu_endpoint_name,
        sagemaker_session=sess,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )
    predicted = predictor.predict(questions)
    return predicted['text'], predicted['intentions'], predicted['slot_labels']


def extract_slot_values(question, seq_label):
    assert len(question) == len(seq_label), f"question {question} should have the same \
length with sequence label {seq_label} ({len(question)} != {len(seq_label)})"
    value_buf = ''
    slot_buf = ''
    values = []
    slots = []
    for i, l in enumerate(seq_label):
        if l.startswith('B'):
            if value_buf != '':
                values.append(value_buf)
                slots.append(slot_buf)
            slot_buf = l[2:] # extract label part from B_label
            value_buf = question[i]
        elif l.startswith('I'):
            value_buf += question[i]
        elif l.startswith('O'):
            if value_buf != '':
                values.append(value_buf)
                slots.append(slot_buf)
            value_buf = ''
            slot_buf = ''  
    return slots, values


def generate_graph_query(intent, slots, values, query_templates):
    if intent not in query_templates.keys():
        raise Exception(f"Query templates does not have a template for {intent}")
    template = query_templates[intent]
    query_expr = template.format(*values)
    return query_expr


def question2answer(question, query_templates, nlu_endpoint_name):
    _, intentions, slot_labels = parse_questions([question], nlu_endpoint_name)
    print(f"Intention: {intentions[0]}")
    slots, values = extract_slot_values(question, slot_labels[0])
    print(f"Slot labels: {slots},{values}")
    query_expr = generate_graph_query(intentions[0], slots, values, query_templates)
    print(f"Query: {query_expr}")
    return query(expr=query_expr)


def prepare_iamdb_request(database_url):

    service = 'neptune-db'
    method = 'GET'

    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    region = os.environ['AWS_REGION']
    session_token = os.environ['AWS_SESSION_TOKEN']

    creds = SimpleNamespace(
        access_key=access_key, secret_key=secret_key, token=session_token, region=region,
    )

    request = AWSRequest(method=method, url=database_url, data=None)
    SigV4Auth(creds, service, region).add_auth(request)

    return httpclient.HTTPRequest(database_url, headers=request.headers.items())


def is_retriable_error(e):

    is_retriable = False
    err_msg = str(e)

    if isinstance(e, tuple(network_errors)):
        is_retriable = True
    else:
        is_retriable = any(
            retriable_err_msg in err_msg for retriable_err_msg in retriable_err_msgs)

    print('error: [{}] {}'.format(type(e), err_msg))
    print('is_retriable: {}'.format(is_retriable))

    return is_retriable


def is_non_retriable_error(e):
    return not is_retriable_error(e)


def reset_connection_if_connection_issue(params):

    is_reconnectable = False

    e = sys.exc_info()[1]
    err_msg = str(e)

    if isinstance(e, tuple(network_errors)):
        is_reconnectable = True
    else:
        is_reconnectable = any(
            reconnectable_err_msg in err_msg for reconnectable_err_msg in reconnectable_err_msgs)

    print('is_reconnectable: {}'.format(is_reconnectable))

    if is_reconnectable:
        global conn
        global g
        conn.close()
        conn = create_remote_connection()
        g = create_graph_traversal_source(conn)


@backoff.on_exception(backoff.constant,
                      tuple(retriable_errors),
                      max_tries=5,
                      jitter=None,
                      giveup=is_non_retriable_error,
                      on_backoff=reset_connection_if_connection_issue,
                      interval=1)
def query(**kwargs):

    expr = kwargs['expr']
    result_code = 'SUCCEEDED'
    try:
        result = eval(expr)
    except Exception as e:
        result_code = 'FAILED'
        result = str(e)
    result_dict = {
        'result_code': result_code,
        'result': result
    }
    return result_dict


def doQuery(event):
    question = event['question']
    with open('query_templates.json') as f:
        query_templates = json.load(f)
    nlu_endpoint_name = os.environ['nluEndpoint']
    return question2answer(question, query_templates, nlu_endpoint_name)


def lambda_handler(event, context):
    return doQuery(event)


def create_graph_traversal_source(conn):
    return traversal().withRemote(conn)


def create_remote_connection():
    print('Creating remote connection')

    return DriverRemoteConnection(
        connection_string(),
        'g',
        pool_size=1,
        message_serializer=serializer.GraphSONSerializersV2d0())


def connection_string():
    port = os.environ['neptunePort']
    if port == 80 or port == '80': # use unencrypted web socket if port is an http port
        database_url = 'ws://{}:{}/gremlin'.format(
            os.environ['neptuneEndpoint'], os.environ['neptunePort'])
    else:
        database_url = 'wss://{}:{}/gremlin'.format(
            os.environ['neptuneEndpoint'], os.environ['neptunePort'])

    if 'USE_IAM' in os.environ and os.environ['USE_IAM'] == 'true':
        return prepare_iamdb_request(database_url)
    else:
        return database_url


conn = create_remote_connection()
g = create_graph_traversal_source(conn)
