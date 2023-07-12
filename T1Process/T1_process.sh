#!/bin/bash

usage="${0} <T1> <output directory> <script dir>"

if [ ${#} -lt 2 ]; then
    echo ${usage}
    exit
fi

printStatus() {
    printf \\e[33m"\tT1: ${1}\n"\\e[0m
}

printWarning() {
    printf \\e[31m"\tT1: ${1}\n"\\e[0m
}


inT1=${1}
T1Dir=${2}
thisDir=${3}


if [ ! -d ${T1Dir} ]; then
    printStatus "Making dir " ${T1Dir}
    mkdir -p ${T1Dir}
fi

stdOut=${T1Dir}/Output.txt
stdErr=${T1Dir}/Errors.txt
echo "-------------`date`---------------" >> ${stdOut}
echo "-------------`date`---------------" >> ${stdErr}

cp ${inT1} ${T1Dir}/anat.nii.gz
inT1=${T1Dir}/anat.nii.gz

fslreorient2std ${inT1} ${inT1} >>${stdOut} 2>>${stdErr}

if [ ! -e ${T1Dir}/anat_BiasCorr_restore.nii.gz ]; then
    printStatus "Bias correction"
    fast -o ${T1Dir}/anat_BiasCorr -l 10 -b -B -t 1 --iter=5 --nopve --fixed=0 -v ${inT1} >>${stdOut} 2>>${stdErr}
fi

#echo "First Segmentation"
#if [ ! -e ${T1Dir}/anat_BiasCorr_restore_all_fast_firstseg.nii.gz ]; then
#    run_first_all -i ${T1Dir}/anat_BiasCorr_restore.nii.gz -o ${T1Dir}/anat_BiasCorr_restore &
#fi

if [ ! -e ${T1Dir}/anat_ss.nii.gz ]; then
    printStatus "BET"
    bet ${T1Dir}/anat_BiasCorr_restore.nii.gz ${T1Dir}/anat_ss -B -f .3 >>${stdOut} 2>>${stdErr}
fi

if [ ! -e ${T1Dir}/highres2standard.mat ]; then
    printStatus "Perform Normalization"
    flirt -ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -in ${T1Dir}/anat_ss.nii.gz -out ${T1Dir}/highres2standard.nii.gz -omat ${T1Dir}/highres2standard.mat >>${stdOut} 2>>${stdErr}
fi

if [ ! -e ${T1Dir}/highres2standard.png ]; then
    printStatus "Generate images"
    slicesmask ${T1Dir}/anat.nii.gz ${T1Dir}/anat_ss.nii.gz ${T1Dir}/anat_ss.png >>${stdOut} 2>>${stdErr} &
    slicesmask ${T1Dir}/highres2standard.nii.gz ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz ${T1Dir}/highres2standard.png >>${stdOut} 2>>${stdErr} &
    slicesmask ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz ${T1Dir}/highres2standard.nii.gz ${T1Dir}/highres2standard2.png >>${stdOut} 2>>${stdErr} &
    slicer ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz ${T1Dir}/highres2standard.nii.gz -S 2 600 ${T1Dir}/MNI152_with_highres2standard_outline.png >>${stdOut} 2>>${stdErr} &
    slicer ${T1Dir}/highres2standard.nii.gz ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -S 2 600 ${T1Dir}/highres2standard_with_MNI152_outline.png >>${stdOut} 2>>${stdErr} &
fi

if [ ! -e ${T1Dir}/seg_seg_0.nii.gz ]; then
    printStatus "Perform Segmentation"
    fast -g -p -o ${T1Dir}/seg ${T1Dir}/anat_ss.nii.gz >>${stdOut} 2>>${stdErr}
fi

if [ ! -e ${T1Dir}/seg_pve_2_thr95_eroded_masked.nii.gz ]; then
    printStatus "Thresholding Segmentations"
    # Threshold pve for WM and CSF at .95
    fslmaths ${T1Dir}/seg_pve_0.nii.gz -thr .95 ${T1Dir}/seg_pve_0_thr95.nii.gz >>${stdOut} 2>>${stdErr}
    fslmaths ${T1Dir}/seg_pve_1.nii.gz -thr .95 ${T1Dir}/seg_pve_1_thr95.nii.gz >>${stdOut} 2>>${stdErr}
    fslmaths ${T1Dir}/seg_pve_2.nii.gz -thr .95 ${T1Dir}/seg_pve_2_thr95.nii.gz >>${stdOut} 2>>${stdErr}
    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_0_thr95.nii.gz -o ${T1Dir}/seg_pve_0_thr95_mask.png >>${stdOut} 2>>${stdErr}
    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_1_thr95.nii.gz -o ${T1Dir}/seg_pve_1_thr95_mask.png >>${stdOut} 2>>${stdErr}
    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_2_thr95.nii.gz -o ${T1Dir}/seg_pve_2_thr95_mask.png >>${stdOut} 2>>${stdErr}

    #Erode WM and CSF masks
    fslmaths ${T1Dir}/seg_pve_0_thr95.nii.gz -kernel 3D -ero ${T1Dir}/seg_pve_0_thr95_eroded.nii.gz >>${stdOut} 2>>${stdErr}
    fslmaths ${T1Dir}/seg_pve_1_thr95.nii.gz -kernel 3D -ero ${T1Dir}/seg_pve_1_thr95_eroded.nii.gz >>${stdOut} 2>>${stdErr}
    fslmaths ${T1Dir}/seg_pve_2_thr95.nii.gz -kernel 3D -ero ${T1Dir}/seg_pve_2_thr95_eroded.nii.gz >>${stdOut} 2>>${stdErr}
    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_0_thr95_eroded.nii.gz -o ${T1Dir}/seg_pve_0_thr95_eroded_mask.png >>${stdOut} 2>>${stdErr}
    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_1_thr95_eroded.nii.gz -o ${T1Dir}/seg_pve_1_thr95_eroded_mask.png >>${stdOut} 2>>${stdErr}
    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_2_thr95_eroded.nii.gz -o ${T1Dir}/seg_pve_2_thr95_eroded_mask.png >>${stdOut} 2>>${stdErr}

    #Bring MNI tissue priors to Subject space
    convert_xfm -omat ${T1Dir}/standard2highres.mat -inverse ${T1Dir}/highres2standard.mat >>${stdOut} 2>>${stdErr}
    flirt -ref ${T1Dir}/anat_ss.nii.gz -in ${FSLDIR}/data/standard/tissuepriors/avg152T1_white -out ${T1Dir}/avg152T1_white_inHighres -applyxfm -init ${T1Dir}/standard2highres.mat >>${stdOut} 2>>${stdErr}
    flirt -ref ${T1Dir}/anat_ss.nii.gz -in ${FSLDIR}/data/standard/tissuepriors/avg152T1_csf -out ${T1Dir}/avg152T1_csf_inHighres -applyxfm -init ${T1Dir}/standard2highres.mat >>${stdOut} 2>>${stdErr}

    fslmaths ${T1Dir}/avg152T1_white_inHighres -thr 200 -bin ${T1Dir}/avg152T1_white_inHighres_200Thr >>${stdOut} 2>>${stdErr}
    fslmaths ${T1Dir}/avg152T1_csf_inHighres -thr 128 -bin ${T1Dir}/avg152T1_csf_inHighres_128Thr >>${stdOut} 2>>${stdErr}

    fslmaths ${T1Dir}/seg_pve_0_thr95_eroded.nii.gz -mas ${T1Dir}/avg152T1_csf_inHighres_128Thr ${T1Dir}/seg_pve_0_thr95_eroded_masked >>${stdOut} 2>>${stdErr}
    fslmaths  ${T1Dir}/seg_pve_2_thr95_eroded.nii.gz -mas ${T1Dir}/avg152T1_white_inHighres_200Thr ${T1Dir}/seg_pve_2_thr95_eroded_masked >>${stdOut} 2>>${stdErr}

    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_0_thr95_eroded_masked.nii.gz -o ${T1Dir}/seg_pve_0_thr95_eroded_masked.png >>${stdOut} 2>>${stdErr}
    slices ${T1Dir}/anat_ss.nii.gz ${T1Dir}/seg_pve_2_thr95_eroded_masked.nii.gz -o ${T1Dir}/seg_pve_2_thr95_eroded_masked.png >>${stdOut} 2>>${stdErr}
fi

if [ ! -e index.html ]; then
    printStatus "Generate HTML"
    python "${thisDir}/gen_html.py" ${T1Dir} 'T1'
fi

printStatus "T1 Analysis Complete"
