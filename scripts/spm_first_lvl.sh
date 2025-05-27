module load matlab
timestamp=$(date +%Y%m%d_%H%M%S)
matlab -nodisplay -nosplash -nodesktop -r \
"run('/home/ubuntu/repos/learning-habits-analysis/matlab/glm1.m'); exit;"