
#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/solver.hpp"
#include "caffe/proto/caffe.pb.h"
#include <mpi.h>
#include <pthread.h>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

using caffe::SyncedMemory;

extern int cp;
/*
namespace caffe {

    template <typename Dtype>
        class DxyDataLayer : public DataLayer<Dtype> {
            public:
                DxyDataLayer(const LayerParameter& param) : DataLayer<Dtype>(param){
                }
                void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top) {
                    DataLayer<Dtype>::DataLayerSetUp(bottom,top);
                }

        };

    INSTANTIATE_CLASS(DxyDataLayer);
    REGISTER_LAYER_CLASS(DxyData);
};
*/
extern int rank;
extern int task_count;
namespace conv {

    size_t getSize(const vector<Blob<float>*>& params, int from, int to) {
        CHECK_GE(from, 0) << "from should be non-negative.";
        CHECK_GE(to, from) << "to should be greater than or equal to from.";
        CHECK_GE(params.size()-1, to) << "to should be less than " << params.size();

        size_t size = 0;
        for (int i = from; i <= to; ++i)
            size += params[i]->count();

        return size;
    }


    void send(const void *data, size_t sz, int to, int tag = 0) {
        MPI_Ssend(data, sz, MPI_FLOAT, to, tag, MPI_COMM_WORLD);
    }
    void recv(void *data, size_t sz, int from, int tag = 0) {
        MPI_Recv(data, sz, MPI_FLOAT, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int bro_rank;   //在同一个显卡中的rank
    char msg[1];
    void signal() {
        MPI_Ssend(msg, 1, MPI_CHAR, bro_rank, 7, MPI_COMM_WORLD);
        LOG(INFO) << "rank " << rank << " signal bro rank " << bro_rank;
    }
    void lock() {
        MPI_Recv(msg, 1, MPI_CHAR, bro_rank, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LOG(INFO) << "rank " << rank << " got signal from bro rank " << bro_rank;
    }


    caffe::SGDSolver<float> *solver;
    caffe::Net<float> *net_;


    //以下代码用于结果到fc
    void sendResult() {
        send(net_->top_vecs()[15][0]->data()->cpu_data(), net_->top_vecs()[15][0]->count(), 0, 6);
        send(net_->top_vecs()[0][1]->data()->cpu_data(), net_->top_vecs()[0][1]->count(), 0, 6);//把label发给fc
    }

    //以下代码用于发送diff到fc
    float *diff;
    size_t dsz;
    pthread_t dtid;
    //将diff结果放到diff中，等待发送
    void saveDiff() {
        float* ptr = diff;
        const vector<Blob<float>*>& blobs = net_->learnable_params();
        for (int i = 0; i < 10; ++i) {  //前10个是conv的参数
            size_t sz = blobs[i]->count();
            CUDA_CHECK(cudaMemcpy(ptr, blobs[i]->diff()->gpu_data(), sz * sizeof(float), cudaMemcpyDefault));
            ptr += sz;
        }
    }
    void *sendDiff(void *para) {
        saveDiff();
        send(diff, dsz, 0, 8);
    }

    //以下代码用于接收fc发的结果
    void waitForResult() {
        recv(net_->top_vecs()[15][0]->diff()->mutable_cpu_data(), net_->top_vecs()[15][0]->count(), 0, 9);
    }

    //以下代码用于接收fc发的参数
    float *param;       //下一次应用的参数
    size_t psz;         //param的总大小
    pthread_t ptid;
    void * waitForParam(void *para) {
        recv(param, psz, 0, 18);   //接收diff，标记为8号信息，由于远程机器是1-6，所以要做一个id转换。
    }

    //将参数更新到GPU中
    void updateParam() {
        float* ptr = param;
        const vector<Blob<float>*>& blobs = net_->learnable_params();
        for (int i = 0; i < 10; ++i) {
            size_t sz = blobs[i]->count();
            CUDA_CHECK(cudaMemcpy(blobs[i]->data()->mutable_gpu_data(), ptr, sz * sizeof(float), cudaMemcpyDefault));
            ptr += sz;
        }
    }

    //以下代码用于初始化所有进程中的参数
    void syncParams() {
        MPI_Bcast(param, psz, MPI_FLOAT, 0, MPI_COMM_WORLD);
        updateParam();
    }

    int device;
    void init() {
        Caffe::SetDevice(device);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(1);

        caffe::SignalHandler signal_handler(caffe::SolverAction::STOP,caffe::SolverAction::SNAPSHOT);

        //初始化Solver
        caffe::SolverParameter solver_param;
        caffe::ReadSolverParamsFromTextFileOrDie("./models/bvlc_reference_caffenet/solver.prototxt", &solver_param);
        solver = new caffe::SGDSolver<float>(solver_param);
        solver->SetActionFunction(signal_handler.GetActionFunction());

        //初始化net
        net_ = solver->net().get();

        // 初始化参数空间
        dsz = getSize(net_->learnable_params(), 0, 9);  //前10个参数
        psz = dsz;
        CUDA_CHECK(cudaMallocHost(&diff, dsz * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&param, psz * sizeof(float)));
    }

    void conv(int gpu, int ibro) {
        device = gpu;
        cp = rank;  //给输入用的东东
        printf("rank: %d\n", rank);
        bro_rank = ibro;


        init();

        syncParams();

        bool first = true;

        while(1) {
            LOG(INFO) << "rank " << rank << " start loop";
            net_->ClearParamDiffs();
            if (first) {
                first = false;
                if (rank > bro_rank) {
                    //pthread_create(&dtid, NULL, sendDiff, NULL);
                    lock();
                }
            }
            pthread_create(&ptid, NULL, waitForParam, NULL);


            net_->ForwardFromTo(0,15);
            signal();//通知兄弟结点执行

            LOG(INFO) << "rank " << rank << " sending data to fc";
            sendResult();

            // calculated in fc

            waitForResult();
            LOG(INFO) << "rank " << rank << " got data from fc";

            lock(); //等待GPU空闲
            net_->BackwardFromTo(15,0);

            pthread_join(dtid, NULL);
            pthread_create(&dtid, NULL, sendDiff, NULL);

            pthread_join(ptid, NULL);
            updateParam();

            LOG(INFO) << "rank " << rank << " end loop";
        }
        LOG(INFO) << "rank " << rank << " dxy end backward";

    }
};
