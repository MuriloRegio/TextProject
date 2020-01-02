


implementations_list = ["EAST_PROCESSOR", "FOTS_PROCESSOR"],
     # File names where the necessary 'predict' functions are implemented.

ims_path="~/datasets/ICDAR15_TEST"
     # Directory where the test images are located.

gts_path="~/datasets/ICDAR15_TEST-GROUND_TRUTH"
     # Directory where the GT for the test images is located (ICDAR format).

ext="jpg"
     # File format of the test images.

thresh=0.5
     # IoU threshold for defining correct region predictions.

outname="results"
     # File name for the test results.

logfile="log"
     # File name for the log file.

dumpfile="dump"
     # File name for the dump file. (intermediate results)
     # Empty string means no dump will be made