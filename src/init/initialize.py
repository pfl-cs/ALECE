import os
import shutil
import pathlib
import sys
import argparse
sys.path.append("../")
from src.utils import file_utils, FileViewer

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Initialize some configuraitons')
    parser.add_argument('--PG_DATADIR', type=str, default=None, help='')
    parser.add_argument('--PG_USER', type=str, default=None, help='')
    args = parser.parse_args()
    return args



def replace_term(path, old_term, new_term):
    lines = file_utils.read_all_lines(path)
    newlines = []
    start_i = -1
    for i, line in enumerate(lines):
        if line.startswith('--'):
            start_i = i
            break
        else:
            newline = line.replace(old_term, new_term)
            newlines.append(newline)
    if start_i >= 0:
        newlines.extend(lines[start_i:])

    file_utils.write_all_lines(path, newlines)


def set_workspace_path(workspace_dir):
    data_dir = os.path.join(workspace_dir, 'data')
    paths = FileViewer.list_files(data_dir, suffix='workload.sql', isdepth=True)
    for path in paths:
        fname = os.path.basename(path)
        if fname == 'workload.sql':
            print(f'path = {path}')
            # replace_term(path, workspace_dir, '$WORKSPACE_DIR$')
            replace_term(path, '$WORKSPACE_DIR$', workspace_dir)

def set_pg_configs(args, workspace_dir):
    assert args.PG_DATADIR is not None and args.PG_USER is not None
    arg_parser_path = os.path.join(workspace_dir, 'src/arg_parser/arg_parser.py')
    replace_term(arg_parser_path, '$PG_DATADIR$', args.PG_DATADIR)

    pg_utils_path = os.path.join(workspace_dir, 'src/utils/pg_utils.py')
    replace_term(pg_utils_path, '$PG_USER$', args.PG_USER)



if __name__ == '__main__':
    args = get_arg_parser()
    workspace_dir = str(pathlib.Path().resolve().parent.absolute())
    if args.PG_DATADIR is None or args.PG_USER is None:
        set_workspace_path(workspace_dir)
    else:
        set_pg_configs(args, workspace_dir)

# python init/initialize.py
# python init/initialize.py --PG_DATADIR /home/admin/lpf_files/pg_data --PG_USER lpf367135

# python init/initialize.py
# python init/initialize.py --PG_DATADIR /apsarapangu/disk1/lpf_files/pg_data --PG_USER lpf367135