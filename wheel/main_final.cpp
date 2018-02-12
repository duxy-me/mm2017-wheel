#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/signal_handler.h"
#include <mpi.h>
#include "main_conv.cpp"
#include "main_fc.cpp"
#include <pthread.h>

int rank;
int task_count;

int main(int argc, char** argv) {
	//google::InitGoogleLogging("VR");
	//FLAGS_log_dir = "/home/dxy/caffe/dxy/log_new";
	//FLAGS_stderrthreshold = google::ERROR;
    
	/*mpi devide*/
	char hostname[MPI_MAX_PROCESSOR_NAME];
	int len;
	int ret;
	int provided;

	ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided != MPI_THREAD_MULTIPLE) {
		printf("MPI do not Support Multiple thread\n");
		return 0;
	}
	if (MPI_SUCCESS != ret) {
		printf("start mpi fail\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &task_count);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(hostname, &len);
	printf("task_count = %d, my rank = %d on %s\n", task_count, rank, hostname);

	int totalDevices = 0;
    int gpu[2];
	cudaGetDeviceCount(&totalDevices);
	for (int i = 0; i < totalDevices; i++) {
		struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		/*
		printf(
				"device %d(%s) : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",
				i, prop.name, prop.multiProcessorCount, prop.major, prop.minor,
				(float) prop.clockRate * 1e-3,
				(int) (prop.totalGlobalMem / (1024 * 1024)),
				(float) prop.memoryClockRate * 1e-3, prop.ECCEnabled,
				prop.multiGpuBoardGroupID);
				*/
		if (strcmp(prop.name, "Tesla K40c") == 0) {
			gpu[1] = gpu[0];
			gpu[0] = i;
		}
	}
	
	if (rank == 0) {
		LOG(INFO) << "rank 0 start at host " << hostname << " gpu " << gpu[0];
		fc::fc(gpu[0]);
	} else if (rank == 1) {
		LOG(INFO) << "rank 1 start at host " << hostname << " gpu " << gpu[1];
		conv::conv(gpu[1], 2);
	}else if (rank == 2) {
		LOG(INFO) << "rank 2 start at host " << hostname << " gpu " << gpu[1];
		conv::conv(gpu[1], 1);
	}else if (rank == 3) {
		LOG(INFO) << "rank 3 start at host " << hostname << " gpu " << gpu[0];
		conv::conv(gpu[0], 4);
	}else if (rank == 4) {
		LOG(INFO) << "rank 4 start at host " << hostname << " gpu " << gpu[0];
		conv::conv(gpu[0], 3);
	}else if (rank == 5) {
		LOG(INFO) << "rank 5 start at host " << hostname << " gpu " << gpu[1];
		conv::conv(gpu[1], 6);
	}else if (rank == 6) {
		LOG(INFO) << "rank 6 start at host " << hostname << " gpu " << gpu[1];
		conv::conv(gpu[1], 5);
	}
	
	
	
	MPI_Finalize();
}

