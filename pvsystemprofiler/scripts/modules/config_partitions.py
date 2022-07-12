class ConfigPartitions:
    def __init__(self, est=None, part_id=None, ix_0=None, ix_n=None, n_part=None, ifl=None, ofl=None, ip_address=None,
                 skf=None, au=None, ain=None, ar=None, ac=None, script_name=None, scripts_location=None, conda_env=None,
                 pcid=None, gof=None, god=None, cts=None, s3l=None, n_files=None, file_label=None, fix_time_shifts=None,
                 time_zone_correction=None, check_json=None, sup_file=None, ds=None, gmt=None):
        self.estimation = est
        self.input_file_location = ifl
        self.ssh_key_file = skf
        self.aws_username = au
        self.aws_instance_name = ain
        self.aws_region = ar
        self.aws_client = ac
        self.supplementary_file = sup_file
        if ix_0 is not None and ix_n is not None:
            self.part_id = part_id
            self.env_name = 'partition_' + str(part_id)
            self.n_part = n_part
            self.local_working_folder_location = ofl
            self.local_working_folder = self.local_working_folder_location + 'estimation_part_{}_of_{}/'.format(
                str(part_id + 1), str(n_part))
            self.local_input_file = self.local_working_folder + 'input_part_{}_of_{}.csv'.format(str(part_id + 1),
                                                                                                 str(n_part))
            self.local_output_file_name = 'results_part_{}_of_{}.csv'.format(str(part_id + 1), str(n_part))
            self.local_output_file = self.local_working_folder + self.local_output_file_name
            self.ix_0 = ix_0
            self.ix_n = ix_n
            self.scripts_location = scripts_location
            self.script_name = self.scripts_location + script_name
            self.conda_environment = conda_env
            self.power_column_id = pcid
            self.convert_to_ts = cts
            self.public_ip_address = ip_address
            self.process_completed = False
            self.s3_location = s3l
            self.n_files = n_files
            self.file_label = file_label
            self.fix_time_shifts = fix_time_shifts
            self.time_zone_correction = time_zone_correction
            self.check_json = check_json
            self.data_source = ds
            self.gmt_offset = gmt
        else:
            self.global_output_directory = god
            self.global_output_file = god + gof


def get_config(est=None, part_id=None, ix_0=None, ix_n=None, n_part=None, ifl=None, ofl=None, ip_address=None, skf=None,
               au=None, ain=None, ar=None, ac=None, script_name=None, scripts_location=None, conda_env=None, pcid=None,
               gof=None, god=None, cts=None, s3l=None, n_files=None, file_label=None, fix_time_shifts=None,
               time_zone_correction=None, check_json=None, sup_file=None, data_source=None, gmt_offset=None):
    if ix_0 is not None and ix_n is not None:
        return ConfigPartitions(est=est, part_id=part_id, ix_0=ix_0, ix_n=ix_n, n_part=n_part, ifl=ifl, ofl=ofl,
                                ip_address=ip_address, skf=skf, au=au, ain=ain, ar=ar, ac=ac, script_name=script_name,
                                scripts_location=scripts_location, conda_env=conda_env, pcid=pcid, cts=cts, s3l=s3l,
                                n_files=n_files, file_label=file_label, fix_time_shifts=fix_time_shifts,
                                time_zone_correction=time_zone_correction, check_json=check_json, sup_file=sup_file,
                                ds=data_source, gmt=gmt_offset)
    else:
        return ConfigPartitions(est=est, ifl=ifl, skf=skf, au=au, ain=ain, ar=ar, ac=ac, gof=gof, god=god)
