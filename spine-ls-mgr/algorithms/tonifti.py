import dicom2nifti
import os

case = "case1"
stu_ins_uid = "1.2.826.0.1.3680043.2.365.1116.112003455.1"
ser_ins_uid = "1.3.12.2.1107.5.2.53.190410.30000024042207074999600000014"

dicom_directory = f"data/{case}/{stu_ins_uid}/{ser_ins_uid}"
output_file = f"data/{case}/series.nii.gz"

dicom2nifti.convert_directory(dicom_directory, os.path.dirname(output_file))