"""

This module provides a class to run the SolarDataTools pipeline on a Dask cluster.
It takes a data plug and a dask client as input and runs the pipeline on the data plug

See the README and example notebook for more information

"""
import os
import sys
import pandas as pd
from time import strftime
from dask import delayed
from dask.distributed import performance_report
from solardatatools import DataHandler


class Runner:
    """ A class to run the SolarDataTools pipeline on a Dask cluster.
        Will handle invalid data keys and failed datasets.

        :param data_plug: a handler to the data plug object.
        :type data_plug: class:`DataPlug`
        :param client: a handler to the Dask client object.
        :type client: class:`Client`
        :param output_path: the path to save the results.
        :type output_path: str
    """

    def __init__(self, data_plug, client, output_path="../results/"):
        self.data_plug = data_plug
        self.client = client
        self.output_path = output_path

    def set_up(self, KEYS, **kwargs):
        """ A function to set up the pipeline on the data plug

        Call run_pipeline functions in a for loop over the keys
        and collect results in a DataFrame

        :param KEYS: List of tuples match dataplug format
        :type KEYS: list
        :param kwargs: additional arguments to pass to the run_pipeline function
        :type kwargs: dict
        """

        
        datas = []

        class Data:
            """ A class to store the results of the pipeline run on a single key
            The object is delayed to be computed.
            """
            reports = None
            loss_reports = None
            runtime = None
            errors = None
            def __init__(self, report, loss_report, runtime, errors):
                self.report = report
                self.loss_report = loss_report
                self.runtime = runtime
                self.errors = errors


        class DaskErrors:
            """ A class to store the errors of the pipeline run on a single key
            """
            get_data_errors = None
            run_pipeline_errors = None
            run_pipeline_report_errors = None
            run_loss_analysis_errors = None
            loss_analysis_report_errors = None
            def __init__(self, get_data_errors, run_pipeline_errors, run_loss_analysis_errors, run_pipeline_report_errors, loss_analysis_report_errors):
                self.get_data_errors = get_data_errors
                self.run_pipeline_errors = run_pipeline_errors
                self.run_pipeline_report_errors = run_pipeline_report_errors
                self.run_loss_analysis_errors = run_loss_analysis_errors
                self.loss_analysis_report_errors = loss_analysis_report_errors


            @staticmethod
            def get_attrs():
                return ["get_data_errors", "run_pipeline_errors", "run_pipeline_report_errors", "run_loss_analysis_errors", "loss_analysis_report_errors"]

        def run(data_plug, key, **kwargs):
            """ A function to get data from dataplug, run sdt pipeline and 
            loss analysis on a single key. Will catch errors and store them in the DaskErrors object.

            :param data_plug: a handler to the data plug object.
            :type data_plug: class:`DataPlug`
            :param key: a key to get data from the data plug
            :type key: tuple
            :param kwargs: additional arguments to pass to the run_pipeline function
            :type kwargs: dict
            """
            report = None
            loss_report = None
            runtime = None
            run_pipeline_error = None
            run_loss_analysis_error = None
            run_pipeline_report_error = None
            loss_analysis_report_error = None

            datahandler = None
            get_data_errors = "No Error"
            try:
                result = data_plug.get_data(key)
                dh = DataHandler(result)
                datahandler = dh
            except Exception as e:
                get_data_errors = str(e)


            if datahandler is None:
                errors = DaskErrors(get_data_errors, "get_data error leading nothing to run", "get_data error leading nothing to analyze", "get_data error leading nothing to report", "get_data error leading nothing to report")
                return Data(report, loss_report, runtime, errors)
            try:
                datahandler.run_pipeline(**kwargs)
                if datahandler.num_days <= 365:
                    run_loss_analysis_error = "The length of data is less than or equal to 1 year, loss analysis will fail thus is not performed."
                    loss_analysis_report_error = "Loss analysis is not performed"

            except Exception as e:
                run_pipeline_error = str(e)
                run_loss_analysis_error = "Failed because of run_pipeline error"
                run_pipeline_report_error = "Failed because of run_pipeline error"
                loss_analysis_report_error = "Failed because of run_pipeline error"
                errors = DaskErrors(get_data_errors, run_pipeline_error, run_pipeline_report_error, run_loss_analysis_error, loss_analysis_report_error)
                return Data(report, loss_report, runtime, errors)

            if run_pipeline_error is None:
                try:
                    report = datahandler.report(return_values=True, verbose=False)
                except Exception as e:
                    run_pipeline_report_error = str(e)
                
                try:
                    runtime = datahandler.total_time
                except Exception as e:
                    print(e)

            if run_loss_analysis_error is None:
                try:
                    datahandler.run_loss_factor_analysis()
                except Exception as e:
                    run_loss_analysis_error = str(e)
                    loss_analysis_report_error = "Failed because of run_loss_analysis error"

                    errors = DaskErrors(get_data_errors, run_pipeline_error, run_pipeline_report_error, run_loss_analysis_error, loss_analysis_report_error)
                    return Data(report, loss_report, runtime, errors)
                
                try:
                    loss_report = datahandler.loss_analysis.report()
                except Exception as e:
                    loss_analysis_report_error = str(e)

            errors = DaskErrors(get_data_errors, run_pipeline_error, run_loss_analysis_error, run_pipeline_report_error, loss_analysis_report_error)
            return Data(report, loss_report, runtime, errors)

        def generate_report(reports, losses):
            """ Generate a dataframe from a list of reports and losses
            """
            
            reports = helper_data(reports)
            losses = helper_data(losses)
            df_reports = pd.DataFrame(reports)
            loss_reports = pd.DataFrame(losses)
            return pd.concat([df_reports, loss_reports], axis=1)

        def helper_data(datas):
            """ Replace None with empty dictionary to create DataFrame
            """
            return [data if data is not None else {} for data in datas]

        def generate_errors(errors):
            """ Generate a dictionary of errors from a list of DaskErrors objects
            """ 
            errors_dict = {}
            for key in DaskErrors.get_attrs():
                errors_dict[key] = []
            for error in errors:
                for key in DaskErrors.get_attrs():
                    err = getattr(error, key)
                    if err is not None:
                        errors_dict[key].append(err)
                    else:
                        errors_dict[key].append("No Error")
            return errors_dict

        
        def prepare_final_report(datas):
            """ A function to generate final summary report, will add runtime and errors to the report

            :param reports: a list of reports, each report is a dictionary
            :type reports: list
            :param losses: a list of losses, each loss is a dictionary
            :type losses: list
            :param runtimes: a list of runtimes, each runtime is a float
            :type runtimes: list
            :param errors: a list of errors, each error is DaskErrors object
            :type errors: list
            """
            
            reports = []
            runtimes = []
            losses = []
            errors = []

            for data in datas:
                reports.append(data.report)
                runtimes.append(data.runtime)
                losses.append(data.loss_report)
                errors.append(data.errors)

            df_reports = generate_report(reports, losses)
            errors_dict = generate_errors(errors)

            columns = {
                "runtime": runtimes,
            }

            for key in DaskErrors.get_attrs():
                columns[key] = errors_dict[key]

            for i in range(len(KEYS[0])):
                columns[f"key_field_{i}"] = [key[i] for key in KEYS]

            return df_reports.assign(**columns)

        # for each key, use Dask delayed to run the pipeline
        for key in KEYS:

            data = delayed(run)(self.data_plug, key, **kwargs)

            datas.append(data)

        self.df_reports = delayed(prepare_final_report)(datas)

    def visualize(self, filename="sdt_graph.png"):
        """ Visualize the pipeline, user should have graphviz installed
        """
        self.df_reports.visualize(filename)

    def get_result(self, dask_report="dask-report.html", summary_report="summary_report.csv"):
        """ Compute the tasks on the cluster and save the results
        """
        time_stamp = strftime("%Y%m%d-%H%M%S")
        # test if the filepath exist, if not create it
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Compute tasks on cluster and save results
        with performance_report(self.output_path + "/" + f"{time_stamp}-" + dask_report):
            summary_table = self.client.compute(self.df_reports)
            df = summary_table.result()
            df.to_csv(self.output_path + "/" + f"{time_stamp}-" + summary_report)

        self.client.shutdown()