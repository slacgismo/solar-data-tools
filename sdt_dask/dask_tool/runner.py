"""

This module provides a class to run the SolarDataTools pipeline on a Dask cluster.
It takes a data plug and a dask client as input and runs the pipeline on the data plug

See the README and tool_demo_SDTDask.ipynb for more information

"""
import os
import sys
import pandas as pd
from time import strftime
from dask import delayed
from dask.distributed import performance_report
from solardatatools import DataHandler


class Runner:
    """A class to run the SolarDataTools pipeline on a Dask cluster.
        Will handle invalid data keys and failed datasets.

        :param keys:
            data_plug (:obj:`DataPlug`): The data plug object.
            client (:obj:`Client`): The Dask client object.
            output_path (str): The path to save the results.
    """

    def __init__(self, data_plug, client, output_path="../results/"):
        self.data_plug = data_plug
        self.client = client
        self.output_path = output_path

    def set_up(self, KEYS, **kwargs):
        """function to set up the pipeline on the data plug

        Call run_pipeline functions in a for loop over the keys
        and collect results in a DataFrame

        :param keys:
            KEYS (list): List of tuples
            **kwargs: Optional parameters.

        """

        reports = []
        runtimes = []
        losses = []
        get_data_errors = []
        errors = []

        class Data:
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
            run_pipeline_errors = None
            run_pipeline_report_errors = None
            run_loss_analysis_errors = None
            loss_analysis_report_errors = None
            def __init__(self, run_pipeline_errors, run_loss_analysis_errors, run_pipeline_report_errors, loss_analysis_report_errors):
                self.run_pipeline_errors = run_pipeline_errors
                self.run_pipeline_report_errors = run_pipeline_report_errors
                self.run_loss_analysis_errors = run_loss_analysis_errors
                self.loss_analysis_report_errors = loss_analysis_report_errors

            @staticmethod
            def get_attrs():
                # return [attr for attr in dir(DaskErrors) if not callable(getattr(DaskErrors,attr)) and not attr.startswith("__")]
                return ["run_pipeline_errors", "run_pipeline_report_errors", "run_loss_analysis_errors", "loss_analysis_report_errors"]


        def run(datahandler, **kwargs):
            run_pipeline_error = None

            if datahandler is None:
                return None
            try:
                datahandler.run_pipeline(**kwargs)
                if datahandler.num_days <= 365:
                    datahandler.run_loss_analysis_error = "The length of data is less than or equal to 1 year, loss analysis will fail thus is not performed."
                    datahandler.loss_analysis_report_error = "Loss analysis is not performed"
                    return datahandler
            except Exception as e:
                datahandler.run_pipeline_error = str(e)
                datahandler.run_loss_analysis_error = "Failed because of run_pipeline error"
                datahandler.run_pipeline_report_error = "Failed because of run_pipeline error"
                datahandler.loss_analysis_report_error = "Failed because of run_pipeline error"
                return datahandler

            try:
                datahandler.run_loss_factor_analysis()
            except Exception as e:
                datahandler.run_loss_analysis_error = str(e)
                datahandler.loss_analysis_report_error = "Failed because of run_loss_analysis error"

                return datahandler

            return datahandler

        def handle_report(datahandler):
            report = None
            loss_report = None
            runtime = None
            run_pipeline_error = None
            run_loss_analysis_error = None
            run_pipeline_report_error = None
            loss_analysis_report_error = None
            if datahandler is None:
                errors = DaskErrors("get_data error leading nothing to run", "get_data error leading nothing to analyze", "get_data error leading nothing to report", "get_data error leading nothing to report")

                return Data(report, loss_report, runtime, errors)

            if hasattr(datahandler, "run_pipeline_error"):
                run_pipeline_error = datahandler.run_pipeline_error
            else:
                try:
                    report = datahandler.report(return_values=True, verbose=False)
                except Exception as e:
                    datahandler.run_pipeline_report_error = str(e)

            if hasattr(datahandler, "run_loss_analysis_error"):
                run_loss_analysis_error = datahandler.run_loss_analysis_error
            else:
                try:
                    loss_report = datahandler.loss_analysis.report()
                except Exception as e:
                    datahandler.loss_analysis_report_error = str(e)

            try:
                runtime = datahandler.total_time
            except Exception as e:
                print(e)

            if hasattr(datahandler, "run_pipeline_report_error"):
                run_pipeline_report_error = datahandler.run_pipeline_report_error

            if hasattr(datahandler, "loss_analysis_report_error"):
                loss_analysis_report_error = datahandler.loss_analysis_report_error

            errors = DaskErrors(run_pipeline_error, run_loss_analysis_error, run_pipeline_report_error, loss_analysis_report_error)
            return Data(report, loss_report, runtime, errors)

        def generate_report(reports, losses):
            reports = helper_data(reports)
            losses = helper_data(losses)
            df_reports = pd.DataFrame(reports)
            loss_reports = pd.DataFrame(losses)
            return pd.concat([df_reports, loss_reports], axis=1)

        def helper_data(datas):
            return [data if data is not None else {} for data in datas]

        def generate_errors(errors):
            # input is a list of Errors objects
            # output is a attribute-list dictionary
            # go through all member in Errors
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

        class DataHandlerData():
            def __init__(self, dh, error):
                self.dh = dh
                self.error = error

        def safe_get_data(data_plug, key):
            datahandler = None
            error = "No Error"
            try:
                result = data_plug.get_data(key)
                dh = DataHandler(result)
                datahandler = dh
            except Exception as e:
                error = str(e)
            return DataHandlerData(datahandler, error)
        
        def prepare_final_report(reports, losses, runtimes, get_data_errors, errors):
            df_reports = generate_report(reports, losses)
            errors_dict = generate_errors(errors)

            columns = {
                "runtime": runtimes,
                "get_data_errors": get_data_errors,
            }

            for key in DaskErrors.get_attrs():
                columns[key] = errors_dict[key]

            for i in range(len(KEYS[0])):
                columns[f"key_field_{i}"] = [key[i] for key in KEYS]

            return df_reports.assign(**columns)


        for key in KEYS:

            dh_data = delayed(safe_get_data)(self.data_plug, key)
            dh_run = delayed(run)(dh_data.dh, **kwargs)
            data = delayed(handle_report)(dh_run)

            reports.append(data.report)
            losses.append(data.loss_report)
            runtimes.append(data.runtime)
            get_data_errors.append(dh_data.error)
            errors.append(data.errors)

        self.df_reports = delayed(prepare_final_report)(reports, losses, runtimes, get_data_errors, errors)

    def visualize(self, filename="sdt_graph.png"):
        # visualize the pipeline, user should have graphviz installed
        self.df_reports.visualize(filename)

    def get_result(self, dask_report="dask-report.html", summary_report="summary_report.csv", additional_columns=pd. DataFrame()):
        # test if the filepath exist, if not create it
        time_stamp = strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # Compute tasks on cluster and save results

        with performance_report(self.output_path + "/" + f"{time_stamp}-" + dask_report):
            summary_table = self.client.compute(self.df_reports)
            df = summary_table.result()
            if not additional_columns.empty:
                df = pd.concat([df, additional_columns], axis=1)
            df.to_csv(self.output_path + "/" + f"{time_stamp}-" + summary_report)

        self.client.shutdown()