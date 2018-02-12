make clean
make mpi_final
scp main im1:`pwd`
cd ..
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64/ -host im2,im1  -np 2 dxy/main
