# wheel

Accelerating CNNs with Distributed GPUs via Hybrid Parallelism and Alternate Strategy

This is our implementation for the paper:

Xiaoyu Du, Jinhui Tang, Zechao Li, and Zhiguang Qin. 2017. [Wheel: Accelerating CNNs with Distributed GPUs via Hybrid Parallelism and Alternate Strategy](https://dl.acm.org/citation.cfm?id=3123435). In Proceedings of the 2017 ACM on Multimedia Conference (MM '17). ACM, New York, NY, USA, 393-401. DOI: https://doi.org/10.1145/3123266.3123435

## Environment Requirements
The codes are for four GPUs deployed on two servers.

We use caffe as the backend.
- caffe
- openMPI with multi-thread support
- cuda
- cudnn

## Execution Example
- `bash mpi.sh`
