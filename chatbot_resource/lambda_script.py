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

reconnectable_err_msgs = [
    'ReadOnlyViolationException',
    'Server disconnected',
    'Connection refused'
]

retriable_err_msgs = [
    'ConcurrentModificationException'] + reconnectable_err_msgs

network_errors = [WebSocketClosedError, OSError]

retriable_errors = [GremlinServerError] + network_errors


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

    id = kwargs['id']

    return (g.V(id)
            .fold()
            .coalesce(
        __.unfold(),
        __.addV('User').property(T.id, id)
    )
        .id().next())


def doQuery(event):
    return query(id=str(randint(0, 10000)))


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

    database_url = 'wss://{}:{}/gremlin'.format(
        os.environ['neptuneEndpoint'], os.environ['neptunePort'])

    if 'USE_IAM' in os.environ and os.environ['USE_IAM'] == 'true':
        return prepare_iamdb_request(database_url)
    else:
        return database_url


conn = create_remote_connection()
g = create_graph_traversal_source(conn)
