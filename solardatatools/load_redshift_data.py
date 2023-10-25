import sshtunnel
import os
import numpy as np
import redshift_connector
from functools import wraps
import pandas as pd
from typing import Callable, TypedDict
from dotenv import load_dotenv
from time import time


class SSHParams(TypedDict):
    ssh_address_or_host: tuple[str, int]
    ssh_username: str
    ssh_private_key: str
    remote_bind_address: tuple[str, int]


class DBConnectionParams(TypedDict):
    database: str
    user: str
    password: str
    host: str
    port: int


load_dotenv()

# SSH connection parameters
ssh_hostname: str = os.environ.get("SSH_HOSTNAME") or ""
ssh_port: int = int(os.environ.get("SSH_PORT") or 22)
ssh_username: str = os.environ.get("SSH_USERNAME") or ""
private_key_file: str = os.environ.get("SSH_PRIVATE_KEY_FILE") or ""

# Redshift connection parameters
db_hostname: str = os.environ.get("DB_HOSTNAME") or ""
db_port: int = int(os.environ.get("DB_PORT") or 5439)
db_name: str = os.environ.get("DB_NAME") or ""
db_user: str = os.environ.get("DB_USER") or ""
db_password: str = os.environ.get("DB_PASSWORD") or ""

# Local host since the tunnel is forwarded to the remote Redshift cluster
db_local_hostname: str = "127.0.0.1"


def create_tunnel_and_connect(ssh_params: SSHParams):
    def decorator(func: Callable):
        @wraps(func)
        def inner_wrapper(db_connection_params: DBConnectionParams, *args, **kwargs):
            with sshtunnel.SSHTunnelForwarder(
                ssh_address_or_host=ssh_params["ssh_address_or_host"],
                ssh_username=ssh_params["ssh_username"],
                ssh_pkey=os.path.abspath(ssh_params["ssh_private_key"]),
                remote_bind_address=ssh_params["remote_bind_address"],
                host_pkey_directories=[
                    os.path.dirname(os.path.abspath(ssh_params["ssh_private_key"]))
                ],
            ) as tunnel:
                if tunnel is None:
                    raise Exception("Tunnel is None")

                tunnel.start()

                if tunnel.is_active is False:
                    raise Exception("Tunnel is not active")

                local_port = tunnel.local_bind_port
                db_connection_params["port"] = local_port

                return func(db_connection_params, *args, **kwargs)

        return inner_wrapper

    return decorator


def load_redshift_data(
    ssh_params: SSHParams,
    redshift_params: DBConnectionParams,
    siteid: str,
    column: str = "ac_power",
    sensor: str | None = None,
    tmin: str | None = None,
    tmax: str | None = None,
    limit: int | None = None,
    verbose: bool = True,
):
    sql_query = """
    SELECT site, meas_name, ts, sensor, meas_val_f FROM measurements
    WHERE site = '{}'
    AND meas_name = '{}'
    """.format(
        siteid, column
    )

    ts_constraint = np.logical_or(tmin is not None, tmax is not None)
    if tmin is not None:
        sql_query += "and ts > '{}'\n".format(tmin)
    if tmax is not None:
        sql_query += "and ts < '{}'\n".format(tmax)
    if sensor is not None and ts_constraint:
        sql_query += "and sensor = '{}'\n".format(sensor)
    elif sensor is not None and not ts_constraint:
        sql_query += "and ts > '2000-01-01'\n"
        sql_query += "and sensor = '{}'\n".format(sensor)
    if limit is not None:
        sql_query += "LIMIT {}".format(limit)
    sql_query += ";"

    @create_tunnel_and_connect(ssh_params)
    def create_df_from_query(redshift_params, sql_query):
        with redshift_connector.connect(**redshift_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                df = cursor.fetch_dataframe()
                return df

    ti = time()
    df = create_df_from_query(redshift_params, sql_query)
    if df is None:
        raise Exception("No data returned from query")
    # df.replace(-999999.0, np.NaN, inplace=True)
    tf = time()
    if verbose:
        print("Query of {} rows complete in {:.2f} seconds".format(len(df), tf - ti))
    return df


if __name__ == "__main__":
    ssh_params: SSHParams = {
        "ssh_address_or_host": (ssh_hostname, ssh_port),
        "ssh_username": ssh_username,
        "ssh_private_key": private_key_file,
        "remote_bind_address": (db_hostname, db_port),
    }

    redshift_params: DBConnectionParams = {
        "database": db_name,
        "user": db_user,
        "password": db_password,
        "host": db_local_hostname,
        "port": 0,
    }
    try:
        df = load_redshift_data(
            ssh_params, redshift_params, siteid="ZT163485000441C1369", limit=10
        )
        if df is None:
            raise Exception("No data returned from query")
        print(df.head(100))
    except Exception as e:
        print(e)
