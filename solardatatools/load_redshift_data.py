import os
import numpy as np
from functools import wraps
from datetime import datetime
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


def load_redshift_data(
    ssh_params: SSHParams,
    redshift_params: DBConnectionParams,
    siteid: str,
    column: str = "ac_power",
    sensor: int | None = None,
    tmin: datetime | None = None,
    tmax: datetime | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Loads data based on a site id from a Redshift database into a Pandas DataFrame using an SSH tunnel

    Parameters
    ----------
        ssh_params : SSHParams
            SSH connection parameters
        redshift_params : DBConnectionParams
            Redshift connection parameters
        siteid : str
            site id to query
        column : str
            meas_name to query  (default ac_power)
        sensor : int, optional
            sensor index to query based on number of sensors at the site id (default None)
        tmin : timestamp, optional
            minimum timestamp to query (default None)
        tmax : timestamp, optional
            maximum timestamp to query (default None)
        limit : int, optional
            maximum number of rows to query (default None)
        verbose : bool, optional
            whether to print out timing information (default False)

    Returns
    ------
    df : pd.DataFrame
        Pandas DataFrame containing the queried data
    """

    try:
        import sshtunnel
    except ImportError:
        raise Exception(
            "Please install sshtunnel into your Python environment to use this function"
        )

    try:
        import redshift_connector
    except ImportError:
        raise Exception(
            "Please install redshift_connector into your Python environment to use this function"
        )

    def timing(verbose: bool = True):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time()
                result = func(*args, **kwargs)
                end_time = time()
                execution_time = end_time - start_time
                if verbose:
                    print(f"{func.__name__} took {execution_time:.2f} seconds to run")
                return result

            return wrapper

        return decorator

    def create_tunnel_and_connect(ssh_params: SSHParams):
        def decorator(func: Callable):
            @wraps(func)
            def inner_wrapper(
                db_connection_params: DBConnectionParams, *args, **kwargs
            ):
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

    @timing(verbose)
    @create_tunnel_and_connect(ssh_params)
    def create_df_from_query(
        redshift_params: DBConnectionParams, sql_query: str
    ) -> pd.DataFrame:
        with redshift_connector.connect(**redshift_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                df = cursor.fetch_dataframe()
                return df

    sensor_found: bool = False
    sensor_dict: dict = {}
    if sensor is not None:
        sensor = sensor - 1

        site_sensor_map_query = f"""
        SELECT sensor FROM measurements
        WHERE site = '{siteid}'
        GROUP BY sensor
        ORDER BY sensor ASC
        """

        site_sensor_df = create_df_from_query(redshift_params, site_sensor_map_query)

        if site_sensor_df is None:
            raise Exception("No data returned from query when getting sensor map")
        sensor_dict = site_sensor_df.to_dict()["sensor"]
        if sensor not in sensor_dict:
            raise Exception(
                f"The index of {sensor + 1} for a sensor at site {siteid} is out of bounds. For site {siteid} please choose a sensor index ranging from 1 to {len(sensor_dict)}"
            )
        sensor_found = True

    sql_query = f"""
    SELECT site, meas_name, ts, sensor, meas_val_f FROM measurements
    WHERE site = '{siteid}'
    AND meas_name = '{column}'
    """

    # ts_constraint = np.logical_or(tmin is not None, tmax is not None)
    if sensor is not None and sensor_found:
        sql_query += f"AND sensor = '{sensor_dict.get(sensor)}'\n"
    if tmin is not None:
        sql_query += f"AND ts > '{tmin}'\n"
    if tmax is not None:
        sql_query += f"AND ts < '{tmax}'\n"
    # if sensor is not None and ts_constraint:
    #     sql_query += f"AND sensor = '{sensor}'\n"
    # elif sensor is not None and not ts_constraint:
    #     sql_query += f"AND ts > '2000-01-01'\n"
    #     sql_query += f"AND sensor = '{sensor}'\n"
    if limit is not None:
        sql_query += f"LIMIT {limit}\n"

    df = create_df_from_query(redshift_params, sql_query)
    if df is None:
        raise Exception("No data returned from query")
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
        start_time = datetime(2017, 8, 2, 19)

        df = load_redshift_data(
            ssh_params,
            redshift_params,
            siteid="ZT163485000441C1369",
            sensor=3,
            tmin=start_time,
            limit=100,
        )
        if df is None:
            raise Exception("No data returned from query")
        if df.empty:
            raise Exception("Empty dataframe returned from query")
        print(df.head(100))
    except Exception as e:
        print(e)
