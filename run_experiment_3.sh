export OMP_NUM_THREADS=1
python ./experiment_3.py 'pardiso_N_5e4_bs_16_sn_false_th1' 
export OMP_NUM_THREADS=2
python ./experiment_3.py 'pardiso_N_5e4_bs_16_sn_false_th2'
#export OMP_NUM_THREADS=3
#python ./experiment_3.py 'pardiso_solve_N_5e4_bs_12_th3'
export OMP_NUM_THREADS=4
python ./experiment_3.py 'pardiso_N_5e4_bs_16_sn_false_th4'
