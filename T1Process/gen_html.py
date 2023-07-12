import sys
import os
import glob


top="""<!DOCTYPE html>
<html>
    <head>
        <link rel="stylesheet" href="style.css">
        <link rel="stylesheet" href="jquery/jquery-ui-1.10.3.custom/css/ui-lightness/jquery-ui-1.10.3.custom.css" />
        <script src="jquery/jquery.js"></script>
        <script src="jquery/jquery-ui-1.10.2/ui/jquery-ui.js"></script>
        <link rel="stylesheet" href="../../../Scripts/style.css" />
        <link rel="stylesheet" type="text/css" href="../../../Scripts/Papaya-master/release/0.7/papaya.css" />
        <script type="text/javascript" src="../../../Scripts/Papaya-master/release/0.7/papaya.js"></script>
        <title>{}</title>
    </head>
    <body>
        <div class="subject_page">


"""

physio = """
        <h2>Raw Data</h2>
        <video width="320" height="240" autoplay loop controls>
          <source src="func.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>


        <h2>Motion</h2>

        <h3>Translation</h3>
        <img src='translation.png' alt="Translation"><br>

        <h3>Rotation</h3>
        <img src='rotation.png' alt="Rotation"><br>

        <h3>Cardiac</h3>
        <a href='cardiac.png'><img src='cardiac.png' alt="cardiac" width="800"></a><br>
        <a href='cardiac_phase.png'><img src='cardiac_phase.png' alt="cardiac" width="800"></a><br>

        <h3>Respiration</h3>
        <a href='respiration.png'><img src='respiration.png' alt="respiration"></a><br>
        <a href='respiration_phase.png'><img src='respiration_phase.png' alt="respiration"></a><br>

        <h2>Outliers</h2>
        <img src='outliers.png' alt="Outliers"><br>

        <h2>Noise Regressors</h2>
        <img src='RS_design.png' alt="Design"><br>
        <a href='RS_design_cov.png' > <img src='RS_design_cov.png' alt="Covariance"></a><br>
        """


noise = """
        <h2>Raw Data</h2>
        <video width="320" height="240" autoplay loop controls>
          <source src="func.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>


        <h2>Motion</h2>

        <h3>Translation</h3>
        <img src='translation.png' alt="Translation"><br>

        <h3>Rotation</h3>
        <img src='rotation.png' alt="Rotation"><br>

        <h2>Outliers</h2>
        <img src='outliers.png' alt="Outliers"><br>

        <h2>Noise Regressors</h2>
        <img src='RS_design.png' alt="Design"><br>
        <a href='RS_design_cov.png' > <img src='RS_design_cov.png' alt="Covariance"></a><br>"""

Stories_results="""
<h1> Results</h1>
<h2>Stories > Tones</h2>
<a href='Stories_Zstats_overlay_p05Z3p6Corrected_StoriesGTTones.png' > <img src='Stories_Zstats_overlay_p05Z3p6Corrected_StoriesGTTones.png' alt="Int>Fix"></a><br>

"""

SDTD_results="""
<h1> Results</h1>
<h2>Animals > Tones</h2>
<a href='SDTD_Zstats_overlay_p05Z3p6Corrected_AnimalsGTTones.png' > <img src='SDTD_Zstats_overlay_p05Z3p6Corrected_AnimalsGTTones.png' alt="2B>0B"></a><br>
"""

T1="""
<h1>BET results</h1>
<a href='anat_ss.png' > <img src='anat_ss.png' alt="anat_ss"></a><br>

<h1>Normalization</h1>
<a href='highres2standard.png' > <img src='highres2standard.png' alt="highres2standard"></a><br>
<a href='highres2standard2.png' > <img src='highres2standard2.png' alt="highres2standard2"></a><br>

"""

end=""" </div>

    </body>

</html>
"""


def run(results_dir, scan_type):
    """docstring for run"""
    filename = os.path.join(results_dir, 'index.html')

    with open(filename, 'w') as fopen:
        fopen.write(top.format(scan_type))
        if scan_type == 'RS':
            fopen.write(physio)
            melodic_images = os.path.join(results_dir, 'rd_rest_mni.ica', 'report', 'IC_*_thresh.png')
            IC_images = glob.glob(melodic_images)
            fopen.write("<h2>Components</h2>")
            IC_images.sort()
            for image in IC_images:
                image_name = os.path.basename(image)
                split_file = os.path.splitext(image_name)[0].split('_')
                fopen.write(" <a href='rd_rest_mni.ica/report/IC_{2}.html'><img class='ic' src='rd_rest_mni.ica/report/{0}' alt='{1}' width='240' height='320'></a>".format(image_name, os.path.splitext(image_name)[0], split_file[1]))

        elif scan_type == 'Stories':
            fopen.write(noise)
            fopen.write(Stories_results)
        elif scan_type == 'SDTD':
            fopen.write(noise)
            fopen.write(SDTD_results)
        elif scan_type == 'T1':
            fopen.write(T1)


        fopen.write(end)





if __name__=="__main__":
    results_dir = sys.argv[1]
    scan_type = sys.argv[2]

    run(results_dir, scan_type)
