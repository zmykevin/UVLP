# make a copy of the code and run from there
# Prevent the interupption on the training due to local code change
save_dir=/data/home/zmykevin/project/local_code/$(date '+%F|%T')
mkdir -p ${save_dir}
cd .. 

cp -u -r mmf-internal ${save_dir}/mmf_code_copy
cd ${save_dir}/mmf_code_copy

echo "Currently in: ${PWD}"
###############Launch the Training Script################
bash launch_pretrain.sh
#########################################################
echo "Finish at dir: ${PWD}"