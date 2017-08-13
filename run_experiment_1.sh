export OMP_NUM_THREADS=1
#python ./experiment_1.py 'scaling_measurement_th1' 1
python ./experiment_1.py 'block_size_scaling_th1' 2
export OMP_NUM_THREADS=2
#python ./experiment_1.py 'scaling_measurement_th2' 1
python ./experiment_1.py 'block_size_scaling_th2' 2
export OMP_NUM_THREADS=3
#python ./experiment_1.py 'scaling_measurement_th3' 1
python ./experiment_1.py 'block_size_scaling_th3' 2
export OMP_NUM_THREADS=4
#python ./experiment_1.py 'scaling_measurement_th4' 1
python ./experiment_1.py 'block_size_scaling_th4' 2
