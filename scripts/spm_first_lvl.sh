module load matlab
matlab -nodisplay -nosplash -nodesktop -r \
"run('/home/ubuntu/repos/learning-habits-analysis/matlab/first_level.m'); exit;" | \
tee matlab_log.txt