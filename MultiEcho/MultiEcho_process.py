import subprocess
from pathlib import Path
import sys
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import align4D
import MultiEcho_exp_fit

import Utils

import shutil
import logging
import pdb


# ==============================================================================
#
# MultiEchoProcess
#
# ==============================================================================
class MultiEchoProcess(object):
    # --------------------------------------------------------------------------
    def __init__(self, echos, anatomical, output_directory, design_mat=None, verbose=False, standard_template='MNI152_T1_2mm_brain.nii.gz',standard_template_mask='MNI152_T1_2mm_brain_mask.nii.gz'):
        """
        Inputs:
        -------
        echos: Object of the MultiEcho class.

        anatomical: Object of the Anat class.

        output_directory: Directory where analysis files will be placed.

        design_mat: Design matrix file if the scan is a task.
                    If resting state then design_mat should be None.

        """

        
        assert(isinstance(echos, MultiEcho))
        self.echos = echos

        assert(isinstance(anatomical, Anat))
        self.anatomical = anatomical

        self.output_directory = Path(output_directory)
        
        self.process_log = self.output_directory.joinpath('processing.log')
        logging.basicConfig(filename=self.process_log, level=logging.DEBUG)
        
        self.verbose = verbose
        self.standard_template = Path(standard_template)
        self.standard_template_mask = Path(standard_template_mask)

        self.design_mat = design_mat
        logging.info(f'Running with design file: {self.design_mat}')
        if self.design_mat:
            self.verify_design()

        # Processed files
        self.coregistered_func = self.output_directory.joinpath("func2highres.nii.gz")
        self.coregistered_mat = self.output_directory.joinpath("func2highres.mat")

        self.normalize_mat = self.output_directory.joinpath('example_func2standard.mat')

        if self.design_mat:
            self.t2_ZStat = self.output_directory.joinpath(self.design_mat.stem + '_T2_ZStats.nii.gz')
            self.t2_COPE = self.output_directory.joinpath(self.design_mat.stem + '_T2_COPES.nii.gz')
            self.t2_VARCOPE = self.output_directory.joinpath(self.design_mat.stem + '_T2_VARCOPES.nii.gz')
            self.t2_weighted_ZStat = self.output_directory.joinpath(self.design_mat.stem + '_T2_weighted_ZStats.nii.gz')
            self.t2_weighted_COPE = self.output_directory.joinpath(self.design_mat.stem + '_T2_weighted_COPES.nii.gz')
            self.t2_weighted_VARCOPE = self.output_directory.joinpath(self.design_mat.stem + '_T2_weighted_VARCOPES.nii.gz')
            

    # --------------------------------------------------------------------------
    def __str__(self):
        #TODO: Print info with the process log and output files along with info on their existence.
        print('')
        
    # --------------------------------------------------------------------------
    def _print(self, message):
        logging.info(message)
        if self.verbose:
            print(message)

    # --------------------------------------------------------------------------
    def process(self):
        self.echos.process()
        self.anatomical.process()
        self.coregistration()
        if self.design_mat:
            self.general_linear_model()

        self.normalize()
        self.report()
        self.gen_info()

    # --------------------------------------------------------------------------
    def verify_design(self):
        """
        Checks that the design file exists and gets the contrast file.
        """
        self._print('Checking design files.')
        Utils.validFile(self.design_mat)
        self.design_con = self.design_mat.with_suffix('.con')
        assert(self.design_con.exists())
        self.design_fsf = self.design_mat.with_suffix('.fsf')
        assert(self.design_fsf.exists())

        local_design_mat = self.output_directory.joinpath(self.design_mat.name)
        shutil.copyfile(self.design_mat, local_design_mat)
        self.design_mat = local_design_mat

        local_design_con = self.output_directory.joinpath(self.design_con.name)
        shutil.copyfile(self.design_con, local_design_con)
        self.design_con = local_design_con

        local_design_fsf = self.output_directory.joinpath(self.design_fsf.name)
        shutil.copyfile(self.design_fsf, local_design_fsf)
        self.design_fsf = local_design_fsf

    # --------------------------------------------------------------------------
    def run_query(self, query, description=None):
        self._print(query)
        if description is not None:
            self._print(f'{description}\n')

        try:
            query_result = subprocess.check_output(query, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            if e.output != b'':
                print(f'\t\t{e.output}')
                logging.error(e.output)
            raise Exception(f"Error running {query[0]}.\n")
        except OSError:
            logging.error(f"Could not find the {query[0]} command.\nError running {query[0]}.\n")
            raise Exception(f"Could not find the {query[0]} command.\nError running {query[0]}.\n")
        except:
            logging.error(f"Error running {query[0]}.\n{' '.join(query)}")
            raise Exception(f"Error running {query[0]}.\n{' '.join(query)}")


    # --------------------------------------------------------------------------
    def coregistration(self):
        """
        Coregister anatomical and functional data.
        """
        logging.info('Running Coregistration')
        if not self.coregistered_mat.exists():
            flirt_query = ["flirt",  "-ref", str(self.anatomical.brain), "-in", str(self.echos.echo3_bet_file), "-out",  str(self.coregistered_func), "-omat", str(self.coregistered_mat)]
            self.run_query(flirt_query, description='Coregistration of functional to anatomical.')
            assert(self.coregistered_func.exists())
            assert(self.coregistered_mat.exists())

    # --------------------------------------------------------------------------
    def general_linear_model(self):
        """
        """
        logging.info('Running GLM')
        
        # Add motion regressors to design
        #TODO - combine motion and outliers into single file and add to design
        feat_query1 = ['feat_model', str(self.design_mat.parent.joinpath(self.design_mat.stem)), str(self.echos.covariates_file)]
        self.run_query(feat_query1, description=f'Add motion parameters to design matrix.')

        if not self.t2_ZStat.exists():
            glm_query1 = ['fsl_glm', '--demean', '-i', str(self.echos.t2_masked_smooth_file), '-d', str(self.design_mat), '-c', str(self.design_con), '-m', str(self.echos.echo3_bet_mask_file), f'--out_z={self.t2_ZStat}', f'--out_cope={self.t2_COPE}', f'--out_varcb={self.t2_VARCOPE}']
            self.run_query(glm_query1, description=f'Run GLM on T2 star file with design {self.design_mat.name}.')
        assert(self.t2_ZStat.exists())
        assert(self.t2_COPE.exists())
        assert(self.t2_VARCOPE.exists())

        if not self.t2_weighted_ZStat.exists():
            glm_query2 = ['fsl_glm', '--demean', '-i', str(self.echos.t2_weighted_masked_smooth_file), '-d', str(self.design_mat), '-c', str(self.design_con), '-m', str(self.echos.echo3_bet_mask_file), f'--out_z={self.t2_weighted_ZStat}', f'--out_cope={self.t2_weighted_COPE}', f'--out_varcb={self.t2_weighted_VARCOPE}']
            self.run_query(glm_query2, description=f'Run GLM on T2 weighted star file with design {self.design_mat.name}.')
        assert(self.t2_weighted_ZStat.exists())
        assert(self.t2_weighted_COPE.exists())
        assert(self.t2_weighted_VARCOPE.exists())

    # --------------------------------------------------------------------------
    def normalize(self):
        """
        Normalize output stat images to standard space.
        """
        logging.info('Running Normalization')
        if not self.normalize_mat.exists():
            xfm_query = ['convert_xfm', '-omat', str(self.normalize_mat), '-concat', str(self.anatomical.highres2standardMat), str(self.coregistered_mat)]
            self.run_query(xfm_query, description=f'Combine transformation matrices.')

        # Used for taking WM and CSF segs to func space for compcor
        # self.highres2func_mat = self.output_directory.joinpath('highres2func.mat')
        # invert_query = ['convert_xfm', '-omat', str(self.highres2func_mat), '-inverse', example_func2highres.mat]
        # self.run_query(invert_query, description=f'')

        #XXX: Normalize smoothed file for resting state!!!
        if self.design_mat:
            for file in [self.t2_ZStat, self.t2_COPE, self.t2_VARCOPE, self.t2_weighted_ZStat, self.t2_weighted_COPE, self.t2_weighted_VARCOPE]:
                norm_out_file = file.parent.joinpath(file.name.replace('.gz', '').replace('.nii', '') + '_mni.nii.gz')
                norm_query = ['flirt', '-in', str(file), '-ref', self.standard_template, '-applyxfm', '-init', self.normalize_mat, '-out', str(norm_out_file)]
                self.run_query(norm_query, description=f'Normalizing {file.name}.')

                mask_query = ['fslmaths', str(norm_out_file), '-mas', str(self.standard_template_mask), str(norm_out_file)]
                self.run_query(mask_query, description=f'Masking {norm_out_file.name}.')


    # --------------------------------------------------------------------------
    def report(self):
        """
        Generate QA images and write process log.
        """
        pass

    # --------------------------------------------------------------------------
    def gen_info(self):
        """
        Generate info.pkl file with processing information.
        """
        pass



# ==============================================================================
#
# MultiEchos
#
# ==============================================================================
class MultiEcho(object):
    # --------------------------------------------------------------------------
    def __init__(self, echo_files, output_directory, echo_times=None, spatial_sigma=2, lowpass_sigma=-1, highpass_sigma=60, verbose=False):

        self.process_log = ''
        self.echo_files = [Path(echo_file) for echo_file in echo_files]
        self.echo_times = echo_times

        self.spatial_sigma = spatial_sigma
        self.lowpass_sigma = lowpass_sigma
        self.highpass_sigma = highpass_sigma

        self.verbose = verbose

        self.output_directory = Path(output_directory)

        self.output_basename = self.output_directory.joinpath('ME_recon')

        ind_echo_sorted = list(np.argsort(self.echo_times))

        self.echo1_file = self.echo_files[ind_echo_sorted.index(0)]#Path(str(self.output_basename) + '_echo1.nii.gz')
        self.echo2_file = self.echo_files[ind_echo_sorted.index(1)]#Path(str(self.output_basename) + '_echo2.nii.gz')
        self.echo3_file = self.echo_files[ind_echo_sorted.index(2)]#Path(str(self.output_basename) + '_echo3.nii.gz')
        self.s0_file = Path(str(self.output_basename) + '_S0.nii.gz')
        self.R2_file = Path(str(self.output_basename) + '_R2.nii.gz')
        self.t2_star_file = Path(str(self.output_basename) + '_T2.nii.gz')
        self.t2_weighted_file = Path(str(self.output_basename) + '_T2weighted.nii.gz')

        self.check_inputs()

        # Processed files
        self.outliers_output = self.output_directory.joinpath("outliers.txt")
        self.split_outliers = []
        self.plot_output = self.output_directory.joinpath("outliers.png")

        self.mc_out_base = self.output_directory.joinpath('Echo3_mcf')
        self.motion_par_file = self.mc_out_base.with_suffix('.par')
        self.motion_meanvol_file = Path(str(self.mc_out_base) + '_meanvol.nii.gz')
        self.motion_mat_directory = self.mc_out_base.with_suffix('.mat')

        self.covariates_file = self.output_directory.joinpath("covariates.txt")

        self.echo3_bet_file = self.output_directory.joinpath('Echo3_mcf_meanvol_brain.nii.gz')
        self.echo3_bet_mask_file = self.output_directory.joinpath('Echo3_mcf_meanvol_brain_mask.nii.gz')

        # Motion corrected
        self.t2_star_mcf_file = Path(str(self.output_basename) + '_T2_mcf.nii.gz')
        self.t2_weighted_mcf_file = Path(str(self.output_basename) + '_T2weighted_mcf.nii.gz')
        self.rotation_plot = self.output_directory.joinpath('rotation.png')
        self.translation_plot = self.output_directory.joinpath('translation.png')

        # Masked
        self.t2_masked_file = Path(str(self.output_basename) + '_T2_mcf_masked.nii.gz')
        self.t2_masked_mean_file = Path(str(self.output_basename) + '_T2_mcf_masked_mean.nii.gz')
        self.t2_masked_std_file = Path(str(self.output_basename) + '_T2_mcf_masked_std.nii.gz')
        self.t2_weighted_masked_file = Path(str(self.output_basename) + '_T2weighted_mcf_masked.nii.gz')
        self.t2_weighted_masked_mean_file = Path(str(self.output_basename) + '_T2weighted_mcf_masked_mean.nii.gz')
        self.t2_weighted_masked_std_file = Path(str(self.output_basename) + '_T2weighted_mcf_masked_std.nii.gz')

        # Smoothed
        self.t2_masked_smooth_file = Path(str(self.output_basename) + '_T2_mcf_masked_smooth.nii.gz')
        self.t2_weighted_masked_smooth_file = Path(str(self.output_basename) + '_T2weighted_mcf_masked_smooth.nii.gz')

    # --------------------------------------------------------------------------
    def __str__(self):
        #TODO: Print info with the process log and output files along with info on their existence.
        print('Not yet implemented!')

    # --------------------------------------------------------------------------
    def check_inputs(self):
        if not self.output_directory.exists():
            self.output_directory.mkdir()

        for echo_file in self.echo_files:
            assert(echo_file.exists())

    # --------------------------------------------------------------------------
    def process(self):
        """
        Extract T2* and weighted echo combination
        """
        if not self.t2_star_file.exists():
            self.echo_times = MultiEcho_exp_fit.run(self.echo_files, str(self.output_basename), echo_times=self.echo_times)

        assert(self.echo1_file.exists())
        assert(self.echo2_file.exists())
        assert(self.echo3_file.exists())
        assert(self.s0_file.exists())
        assert(self.R2_file.exists())
        assert(self.t2_star_file.exists())
        assert(self.t2_weighted_file.exists())

        self.detect_outliers()
        self.cal_motion_correction()
        self.brain_extraction()
        self.apply_motion_correction()

        # Combine motion and outliers into covariates file
        if not self.covariates_file.exists():
            combine_query = ['paste', str(self.motion_par_file), str(self.outliers_output)]
            combined = self.run_query(combine_query, description='Combine motion and outliers into a single file.')
            combined = combined.decode('utf-8')
            with open(self.covariates_file, 'w') as fopen:
                fopen.write(combined)

        #TODO: Need to make filter params an input!
        self.mask()
        self.smooth()

    # --------------------------------------------------------------------------
    def run_query(self, query, description=None):

        #TODO: Add query, description, result and any errors to the process log
        if self.verbose:
            print(f'{description}\n')

        try:
            query_result = subprocess.check_output(query, stderr=subprocess.STDOUT)
            return query_result
        except subprocess.CalledProcessError as e:
            if e.output != b'':
                #print(f'\t\t{e.stderr}')#print(f'\t\t{e.stdout}')
                print(f'\t\t{e.output}')
            raise Exception(f"Error running {query[0]}.\n")
        except OSError:
            raise Exception(f"Could not find the {query[0]} command.\nError running {query[0]}.\n")
        except:
            raise Exception(f"Error running {query[0]}.\n{' '.join(query)}")

    # --------------------------------------------------------------------------
    def brain_extraction(self):
        """
        """
        assert(self.motion_meanvol_file.exists())
        if not self.echo3_bet_file.exists():
            bet_query = ['bet', str(self.motion_meanvol_file), str(self.echo3_bet_file), '-F']
            self.run_query(bet_query, description='Brain extraction of third echo mean volume.')

        assert(self.echo3_bet_file.exists())
        assert(self.echo3_bet_mask_file.exists())

    # ----------------------------------------------------------------------
    def detect_outliers(self):
        if not self.outliers_output.exists():
            outliers_query = ["fsl_motion_outliers", "-i", str(self.echo3_file), "-o", str(self.outliers_output), "-p", str(self.plot_output)]
            self.run_query(outliers_query, description='Calculate outliers based on the third echo.')

        assert(self.outliers_output.exists())

            # Split outliers
            # with open(self.outliers_output, 'r') as fopen:
            #     lines = fopen.readlines()
            #
            # lines = [line.strip().split('  ') for line in lines]
            # for item in range(0, len(lines[0])):
            #     this_outlier_file = self.output_directory.joinpath(f'outlier_{item+1}.txt')
            #     with open(this_outlier_file, 'w') as fopen:
            #         for i in range(0,len(lines)):
            #             fopen.write(lines[i][item]+'\n')
            #     self.split_outliers.append(this_outlier_file)

    # --------------------------------------------------------------------------
    def cal_motion_correction(self):
        """
        """

        if not self.motion_mat_directory.is_dir():
            mc_query = ['mcflirt', '-in', str(self.echo3_file), '-o', str(self.mc_out_base), '-stats', '-mats', '-plots']
            self.run_query(mc_query, description='Motion correction of third echo.')

            mc_plots_query1 = ['fsl_tsplot', '-i', str(self.motion_par_file), '-o', str(self.rotation_plot), '--start=1', '--finish=3', "--title='Rotations (radians)'", '-a', 'x,y,z', '-w', '640', '-h', '144']
            self.run_query(mc_plots_query1, description='Plot rotation parameters.')
            mc_plots_query2 = ['fsl_tsplot', '-i',  str(self.motion_par_file), '-o', str(self.translation_plot), '--start=4', '--finish=6', "--title='Translations (mm)'", '-a', 'x,y,z', '-w', '640', '-h', '144']
            self.run_query(mc_plots_query2, description='Plot translation parameters.')

        assert(self.motion_par_file.exists())
        assert(self.motion_meanvol_file.exists())
        assert(self.motion_mat_directory.is_dir())



    # --------------------------------------------------------------------------
    def apply_motion_correction(self):
        """
        """

        if not self.t2_star_mcf_file.exists():
            apply_mc_query_t2 = ['applyxfm4D', str(self.t2_star_file), str(self.motion_meanvol_file), str(self.t2_star_mcf_file), str(self.motion_mat_directory), '-fourdigit']
            self.run_query(apply_mc_query_t2, description='Apply motion correction to t2 star image.')
        assert(self.t2_star_mcf_file.exists())

        if not self.t2_weighted_mcf_file.exists():
            apply_mc_query_t2weighted = ['applyxfm4D', str(self.t2_weighted_file), str(self.motion_meanvol_file), str(self.t2_weighted_mcf_file), str(self.motion_mat_directory), '-fourdigit']
            self.run_query(apply_mc_query_t2weighted, description='Apply motion correction to the weighted t2 star image.')
        assert(self.t2_weighted_mcf_file.exists())

    # --------------------------------------------------------------------------
    def mask(self):
        if not self.t2_masked_file.exists():
            mask_query = ['fslmaths', str(self.t2_star_mcf_file), '-mas', str(self.echo3_bet_mask_file), str(self.t2_masked_file)]
            self.run_query(mask_query, description='Masking T2 star.')
        if not self.t2_masked_mean_file.exists():
            mean_query = ['fslmaths', str(self.t2_masked_file), '-Tmean', str(self.t2_masked_mean_file)]
            self.run_query(mean_query, description='Mean of T2 star.')
        if not self.t2_masked_std_file.exists():
            std_query = ['fslmaths', str(self.t2_masked_file), '-Tstd', str(self.t2_masked_std_file)]
            self.run_query(std_query, description='Standard deviation of T2 star.')

        if not self.t2_weighted_masked_file.exists():
            mask_query = ['fslmaths', str(self.t2_weighted_mcf_file), '-mas', str(self.echo3_bet_mask_file), str(self.t2_weighted_masked_file)]
            self.run_query(mask_query, description='Masking weighted T2 star.')
        if not self.t2_weighted_masked_mean_file.exists():
            mean_query = ['fslmaths', str(self.t2_weighted_masked_file), '-Tmean', str(self.t2_weighted_masked_mean_file)]
            self.run_query(mean_query, description='Mean of weighted T2 star.')
        if not self.t2_weighted_masked_std_file.exists():
            std_query = ['fslmaths', str(self.t2_weighted_masked_file), '-Tstd', str(self.t2_weighted_masked_std_file)]
            self.run_query(std_query, description='Standard deviation of weighted T2 star.')

    # --------------------------------------------------------------------------
    def smooth(self):
        """
        """
        if not self.t2_masked_smooth_file.exists():
            smooth_query = ['fslmaths', str(self.t2_masked_file), '-s', str(self.spatial_sigma), '-bptf', str(self.highpass_sigma), str(self.lowpass_sigma), str(self.t2_masked_smooth_file)]
            self.run_query(smooth_query, description=f'Apply a Gaussian kernel with sigma {self.spatial_sigma} mm and a band pass temporal filter with high pass of {self.highpass_sigma} volumes and low pass of {self.lowpass_sigma} volumes.')
        assert(self.t2_masked_smooth_file.exists())

        if not self.t2_weighted_masked_smooth_file.exists():
            smooth_query = ['fslmaths', str(self.t2_weighted_masked_file), '-s', str(self.spatial_sigma), '-bptf', str(self.highpass_sigma), str(self.lowpass_sigma), str(self.t2_weighted_masked_smooth_file)]
            self.run_query(smooth_query, description=f'Apply a Gaussian kernel with sigma {self.spatial_sigma} mm and a band pass temporal filter with high pass of {self.highpass_sigma} volumes and low pass of {self.lowpass_sigma} volumes.')
        assert(self.t2_weighted_masked_smooth_file.exists())



# ==============================================================================
#
# Anat
#
# ==============================================================================
class Anat(object):
    # --------------------------------------------------------------------------
    def __init__(self, anatomical_file, output_directory, script_directory, lesion_mask=None, standard_template=Path('MNI152_T1_2mm_brain.nii.gz'), verbose=False):
        self.script_directory = script_directory
        self.anatomical_file = Utils.validFile(anatomical_file)
        self.output_directory = Path(output_directory)

        self.verbose = verbose

        self.brain_mask = self.output_directory.joinpath('anat_ss_mask.nii.gz')
        self.brain = self.output_directory.joinpath('anat_ss.nii.gz')
        self.highres2standardMat = self.output_directory.joinpath('highres2standard.mat')

    # --------------------------------------------------------------------------
    def __str__(self):
        #TODO: Print info with the process log and output files along with info on their existence.
        print('Not yet implemented!')

    # --------------------------------------------------------------------------
    def process(self):
        """

        """
        if not self.highres2standardMat.exists():
            T1_query = [self.script_directory.joinpath('T1_process.sh'), str(self.anatomical_file), str(self.output_directory), str(self.script_directory)]
            self.run_query(T1_query, description='T1 processing.')

        assert(self.brain_mask.exists())
        assert(self.brain.exists())
        assert(self.highres2standardMat.exists())

    # --------------------------------------------------------------------------
    def run_query(self, query, description=None):
        #TODO: Add query, description, result and any errors to the process log
        if self.verbose:
            print(f'{description}\n')

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



# ==============================================================================
if __name__ == "__main__":

    Utils.check_python_version(min_major=3, min_minor=6)
    Utils.check_fsl()

    script_directory = Path(sys.argv[0]).parent.absolute()

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--outdir', metavar='outputDir', type=str, required=True, help="Output directory.")
    parser.add_argument('-a', '--anat', metavar='Anat_file', type=Utils.validFile, required=True, help="Anatomical file or pre-processed directory.")
    parser.add_argument('-d', '--designmat', metavar='DesignMatrix', type=Utils.validFile, required=False, help="Design matrix.")
    parser.add_argument('-i', '--input', metavar='input_files', type=Utils.validFile, required=True, nargs='+', help='Multi-echo input file(s).')
    parser.add_argument('-t', '--te', metavar='echoTimes', type=float, required=False, default=None, help=" Echo times. E.g. 10 20 30", nargs='+')
    parser.add_argument('-l', '--lesion', metavar='lesion_file', type=Utils.validFile, required=False, default=None, help='Lesion mask file [optional].')
    parser.add_argument('-lp', '--lowpass', metavar='lowpass', type=float, required=False, default=-1, help='Low pass temporal filter cutoff in volumes, to not use enter values <0.')
    parser.add_argument('-hp', '--highpass', metavar='highpass', type=float, required=False, default=60, help='High pass temporal filter cutoff in volumes, to not use enter values <0.')
    parser.add_argument('-s', '--sigma', metavar='sigma', type=float, required=False, default=2, help='Sigma (mm) for Gaussian spatial filtering kernel.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output.')
    #TODO: add dummy scan option

    args = parser.parse_args()

    these_echos = MultiEcho(args.input, args.outdir, echo_times=args.te, spatial_sigma=args.sigma, lowpass_sigma=args.lowpass, highpass_sigma=args.highpass, verbose=args.verbose)

    if Path(args.anat).is_dir():
        t1_outdir = Path(args.anat)
        anat_file = t1_outdir.joinpath('anat.nii.gz')
    else:
        anat_file = Path(args.anat)
        t1_outdir = Path(args.outdir).joinpath('T1')
    assert anat_file.exists()
    this_anat = Anat(anat_file, t1_outdir, script_directory, lesion_mask=args.lesion, standard_template=script_directory.joinpath('MNI152_T1_2mm_brain.nii.gz'), verbose=args.verbose)

    this_analysis = MultiEchoProcess(these_echos, this_anat, args.outdir, design_mat=args.designmat, verbose=args.verbose, standard_template=script_directory.joinpath('MNI152_T1_2mm_brain.nii.gz'), standard_template_mask=script_directory.joinpath('MNI152_T1_2mm_brain_mask.nii.gz'))
    this_analysis.process()
