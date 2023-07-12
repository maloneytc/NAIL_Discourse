import subprocess
from pathlib import Path
import sys
import os

import pdb

# ------------------------------------------------------------------------------
def strip_nii_extensions(file_path):
    if file_path.suffixes == ['.nii', '.gz']:
        new_path = Path(str(file_path).replace('.gz', '').replace('.nii', ''))
        return new_path
    elif file_path.suffixes == ['.nii']:
        new_path = Path(str(file_path).replace('.nii', ''))
        return new_path

    raise Exception(f'Unrecognized path {file_path}')


# ------------------------------------------------------------------------------
def check_python_version(min_major=3, min_minor=6):
    """
    Check python version and exit if it does not meet the minimum required version.
    """

    min_python_version = (min_major, min_minor)
    if sys.version_info < min_python_version:
        sys.exit("Python %s.%s or later is required.\n" % min_python_version)

# ------------------------------------------------------------------------------
def check_fsl():
    """
    Check that FSL is installed.
    """
    try:
        fsldir = Path(os.environ["FSLDIR"])
    except KeyError:
        sys.stderr("FSL not found, please install FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) or make sure FSLDIR environment variable is set.")
        sys.exit(1)

# ------------------------------------------------------------------------------
def validFile(inFile):
    """
    If the file exists then it is returned, if not then an exception is raised.
    """
    inFile = Path(inFile)
    if inFile.exists():
        return inFile
    else:
        raise Exception("Could not locate the file: {}".format(inFile))

# ------------------------------------------------------------------------------
def run_query(self, query):
    try:
        query_result = subprocess.check_output(query, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if e.output != b'':
            #print(f'\t\t{e.stderr}')#print(f'\t\t{e.stdout}')
            print(f'\t\t{e.output}')
        raise Exception(f"Error running {query[0]}.\n")
    except OSError:
        raise Exception(f"Could not find the {query[0]} command.\nError running {query[0]}.\n")
    except:
        raise Exception(f"Error running {query[0]}.\n{' '.join(query)}")
