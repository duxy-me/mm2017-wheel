
#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/sgd_solvers.hpp"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

#include "caffe/util/signal_handler.h"
#include <mpi.h>
#include <pthread.h>

#include <algorithm>
#include <string>
#include <vector>

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

extern int rank;
extern int task_count;
namespace caffe {
    template <typename Dtype>
        class DxyDataLayer : public Layer<Dtype> {

            public:
                DxyDataLayer(const LayerParameter& param)
                    : Layer<Dtype>(param){
                    }
                void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
                    Reshape(bottom,top);

                }

                void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
                    printf("%lu %lu\n", top[0]->count(), top[1]->count());
                    int d[4] = {256,3,227,227};
                    vector<int> dims(d,d+4);
                    top[0]->Reshape(dims);

                    vector<int> dims_label;
                    dims_label.push_back(d[0]);
                    top[1]->Reshape(dims_label);
                }
                inline const char* type() const { return "DxyData"; }
                void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
                }

                void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
                }
                void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
                }

        };
    INSTANTIATE_CLASS(DxyDataLayer);
    REGISTER_LAYER_CLASS(DxyData);

};
namespace fc {

    int frank, brank;

    size_t getSize(const vector<Blob<float>*>& params, int from, int to) {
        CHECK_GE(from, 0) << "from should be non-negative.";
        CHECK_GE(to, from) << "to should be greater than or equal to from.";
        CHECK_GE(params.size()-1, to) << "to should be less than " << params.size();

        size_t size = 0;
        for (int i = from; i <= to; ++i)
          size += params[i]->count();

        return size;
    }

    class DxySolver : public caffe::SGDSolver<float> {
        public:
            explicit DxySolver(const caffe::SolverParameter& param)
                : SGDSolver<float>(param) {  }
            explicit DxySolver(const string& param_file)
                : SGDSolver<float>(param_file) {  }
            virtual inline const char* type() const { return "Dxy"; }

            void dxyite(int ite) {
                this->iter_ = ite;
            }
            void Update() {
                //this->net_->Update();
                /* debug
                 const vector<Blob<float>*>& blobs = this->net_->learnable_params();
                 printf("param size: %d\n", blobs.size());
                 float * a;
                 for (int i = 0; i < 16; ++i) {
                     CUDA_CHECK(cudaDeviceSynchronize());
                     //blobs[i]->Update();
                     size_t sz = blobs[i]->count();
                     CUDA_CHECK(cudaMallocHost(&a, sz * sizeof(float)));
                     CUDA_CHECK(cudaMemcpy(a, blobs[i]->diff()->gpu_data(), sz*sizeof(float), cudaMemcpyDeviceToHost));
                     CUDA_CHECK(cudaMemcpy(blobs[i]->data()->mutable_gpu_data(), a, sz*sizeof(float), cudaMemcpyHostToDevice));


                     float* X = (float*) blobs[i]->diff()->mutable_gpu_data();
                     float* Y = (float*) blobs[i]->data()->mutable_gpu_data();
                     float alpha = -1.0;
                     
                     printf(">>>>>>>>  calculate\n");
                     CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), sz, &alpha, X, 1, Y, 1));
                     printf(">>>>>>>>  done calculate\n");
                     //caffe::caffe_gpu_axpy<float>(blobs[i]->count(), -1.0,
                     //           (float*) blobs[i]->diff()->gpu_data(),
                     //           (float*) blobs[i]->data()->mutable_gpu_data());
                 }
                 */
                
                this->ApplyUpdate();
            }
            void dxytest() {
                this->TestAll();
            }
            void dxysave() {
                string fn = this->SnapshotToBinaryProto();
                LOG(INFO) << "saved to " << fn;
            }

    };




    int bro_rank;   //在同一个显卡中的rank
    char msg[1];

    void send(const void *data, size_t sz, int to, int tag = 0) {
        MPI_Ssend(data, sz, MPI_FLOAT, to, tag, MPI_COMM_WORLD);
    }
    void recv(void *data, size_t sz, int from, int tag = 0) {
        MPI_Recv(data, sz, MPI_FLOAT, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    struct data_t {
        float *p;
        size_t sz;
    };
    struct conv_worker_t {
        data_t result;//conv的正向结果
        data_t label; //conv的label
        data_t bp;    //fc的反向残差    
        data_t diff;  //网络diff，包含conv与fc
        data_t param; //使用的参数，包含conv与fc的。其中conv的部分要发给worker
    };

    
    data_t new_param;
    pthread_mutex_t m_param;    //新参数的锁
    conv_worker_t worker[6];
    size_t conv_psz, fc_psz;

    //以下代码用于初始化fc
    int device;
    DxySolver *solver;
    Net<float> *net_;

    DxySolver *pcenter;
    Net<float> *pnet;
    pthread_mutex_t m_pop[6];  //资源的锁
    int current_pop;//当前处理的资源

    //保存param
    void saveParam(float *param, Net<float> *net) {
        float* ptr = param;
        const vector<Blob<float>*>& blobs = net->learnable_params();
        for (int i = 0; i < 16; ++i) {
            size_t sz = blobs[i]->count();
            CUDA_CHECK(cudaMemcpy(ptr, blobs[i]->data()->gpu_data(), sz * sizeof(float), cudaMemcpyDeviceToHost));
            ptr += sz;
        }
    }
    
    
    //以下代码用于载入参数
    void loadParam(float *param, Net<float> *net) {
        float* ptr = param;
        const vector<Blob<float>*>& blobs = net->learnable_params();
        //printf("param size: %d\n", blobs.size());
        for (int i = 0; i < 16; ++i) {
            size_t sz = blobs[i]->count();
            CUDA_CHECK(cudaMemcpy(blobs[i]->data()->mutable_gpu_data(), ptr, sz * sizeof(float), cudaMemcpyDefault));
            ptr += sz;
        }
    }
    
    //以下代码用于载入diff
    void setDiff(float *diff) {
        Net<float> *net = pnet;
        float* ptr = diff;
        const vector<Blob<float>*>& blobs = net->learnable_params();
        for (int i = 0; i < 16; ++i) {
            size_t sz = blobs[i]->count();
            CUDA_CHECK(cudaMemcpy(blobs[i]->diff()->mutable_gpu_data(), ptr, sz * sizeof(float), cudaMemcpyDefault));
            ptr += sz;
        }
    }
    
    //以下代码用于保存fc部分的diff，留待与conv同步更新
    void saveFcDiff(float *fcdiff) {
        Net<float> *net = net_;
        float* ptr = fcdiff;
        const vector<Blob<float>*>& blobs = net->learnable_params();
        for (int i = 10; i < 16; ++i) {
            size_t sz = blobs[i]->count();
            CUDA_CHECK(cudaMemcpy(ptr, blobs[i]->diff()->gpu_data(), sz * sizeof(float), cudaMemcpyDefault));
            ptr += sz;
        }
    }

    void init() {
        CUDA_CHECK(cudaSetDevice(device));
        Caffe::SetDevice(device);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(1);

        size_t newHeapSize = 1024*1024*1024;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeapSize));

        caffe::SignalHandler signal_handler(caffe::SolverAction::STOP,caffe::SolverAction::SNAPSHOT);


        //初始化Solver
        caffe::SolverParameter solver_param;
        caffe::ReadSolverParamsFromTextFileOrDie("./models/bvlc_reference_caffenet/solver_fc.prototxt", &solver_param);
        solver = new DxySolver(solver_param);
        solver->SetActionFunction(signal_handler.GetActionFunction());
        //solver->Restore("./models/bvlc_reference_caffenet/caffenet_train_iter_10000.caffemodel");

        //初始化net
        net_ = solver->net().get();
        net_->CopyTrainedLayersFrom("./models/bvlc_reference_caffenet/mine/caffenet_train_iter_100000.caffemodel");


                //alexnet 参数顺序
        //34848,96,307200,256,884736,384,663552,384,442368,256,37748736,4096,16777216,4096,4096000,1000

        //初始化大小
        conv_psz = getSize(net_->learnable_params(), 0, 9); //前10层是conv的。
        fc_psz = getSize(net_->learnable_params(), 10, 15); //后几层是fc的。

        //初始化全局参数空间
        new_param.sz = conv_psz + fc_psz;
        CUDA_CHECK(cudaMallocHost(&new_param.p, new_param.sz * sizeof(float)));
        saveParam(new_param.p, net_);//把值给放进去
        pthread_mutex_init(&m_param, NULL);

        for (int i=0; i<6; i++) {
            //初始化diff空间
            worker[i].diff.sz = conv_psz + fc_psz;
            CUDA_CHECK(cudaMallocHost(&worker[i].diff.p, worker[i].diff.sz * sizeof(float)));
            memset(worker[i].diff.p, 0, worker[i].diff.sz * sizeof(float));

            //初始化param空间
            worker[i].param.sz = conv_psz + fc_psz;
            CUDA_CHECK(cudaMallocHost(&worker[i].param.p, worker[i].param.sz * sizeof(float)));
            memcpy(worker[i].param.p, new_param.p, worker[i].param.sz * sizeof(float));


            //初始化正向结果空间
            worker[i].result.sz = net_->top_vecs()[15][0]->count();
            CUDA_CHECK(cudaMallocHost(&worker[i].result.p, worker[i].result.sz * sizeof(float)));

            //初始化label空间
            worker[i].label.sz = net_->top_vecs()[0][1]->count();
            CUDA_CHECK(cudaMallocHost(&worker[i].label.p, worker[i].label.sz * sizeof(float)));

            //初始化反向残差空间
            worker[i].bp.sz = net_->bottom_vecs()[16][0]->count();
            CUDA_CHECK(cudaMallocHost(&worker[i].bp.p, worker[i].bp.sz * sizeof(float)));  
        }

        //单独建立一个solver来做参数更新
        pcenter = new DxySolver(solver_param);
        pnet = pcenter->net().get();
        loadParam(new_param.p, pnet);//把值给放进去
        for (int i=0; i<6; i++) {
            pthread_mutex_init(m_pop+i, NULL);
            pthread_mutex_lock(m_pop+i);//完成接收后，抢夺gpu资源，更新
        }
        current_pop = 0;
    }


    //以下代码用于接收conv的正向结果和label
    pthread_t rtid[6];  //6个接收线程
    void * waitForResult(void * remote) {
        int id = (long)remote;
        recv(worker[id].result.p, worker[id].result.sz, id+1, 6);
        recv(worker[id].label.p, worker[id].label.sz, id+1, 6);
    }

    //以下代码用于发送反向的结果
    pthread_t srtid[6];
    void * sendResult(void * remote) {
        int id = (long)remote;
        send(worker[id].bp.p, worker[id].bp.sz, id+1, 9);
    }

    //以下代码用于发送新的参数
    pthread_t sptid[6];
    void * sendParam(void * remote) {
        int id = (long)remote;
        send(worker[id].param.p, conv_psz, id+1, 18);
    }


    
    //以下代码用于接收conv发的diff
    pthread_t dtid[6];  //6个接收线程
    void * waitForDiff(void * remote) {
        int id = (long)remote;
        recv(worker[id].diff.p, conv_psz, id+1, 8);   //接收diff，标记为8号信息，由于远程机器是1-6，所以要做一个id转换。
        pthread_mutex_unlock(m_pop+id);//完成更新，解锁
    }
    //开启Diff接收进程
    void startDiffReceiver() {
        for(int i=0; i<task_count-1; i++) {
            pthread_create(dtid + i, NULL, waitForDiff, (void*)i);
        }
    }
    //开启结果接收进程
    void startResultReceiver() {
        for(int i=0; i<task_count-1; i++) {
            pthread_create(rtid + i, NULL, waitForResult, (void*)i);
        }
    }

    
    //将result和label结果放到net的gpu中
    void inputData(void *cr, void *cl, int r) {
        CUDA_CHECK(cudaMemcpy(cr, worker[r].result.p, worker[r].result.sz * sizeof(float), cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(cl, worker[r].label.p, worker[r].label.sz * sizeof(float), cudaMemcpyDefault));
    }

    //保存残差
    void outputData(const void *output, int r) {
        CUDA_CHECK(cudaMemcpy(worker[r].bp.p, output, worker[r].bp.sz * sizeof(float), cudaMemcpyDefault));
    }

    //以下代码用于初始化所有进程中的参数
    void syncParams() {
        //saveParam(new_param.p);
        MPI_Bcast(new_param.p, conv_psz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    void displayDiff() {
        const vector<Blob<float>*>& blobs = net_->learnable_params();
        printf("----------------------\n");
        for (int i = 0; i < 16; ++i) {
            size_t sz = blobs[i]->count();
            float *ptr = (float*)blobs[i]->diff()->cpu_data();
            for (int j=0; j<7; j++) {
                printf("%f ", ptr[j]);
            }
            printf("\n");
        }
    }
    void displayParam() { 
        const vector<Blob<float>*>& blobs = net_->learnable_params();
        printf("----------------------\n");
        for (int i = 0; i < 16; ++i) {
            size_t sz = blobs[i]->count();
            float *ptr = (float*)blobs[i]->data()->cpu_data();
            for (int j=0; j<7; j++) {
                printf("%f ", ptr[j]);
            }
            printf("\n");
        }
    } 
    void displayLabel() {
        size_t cnt = net_->top_vecs()[0][1]->count();
        float * ptr = (float*)net_->top_vecs()[0][1]->data()->cpu_data();
        for (int i=0; i<cnt; i++) printf("%f ", ptr[i]);;
        printf("\n");
    }

    
    //以下代码用于更新主参数
    //修改版本：以下代码用于将pnet中的参数分发到相应的结果中。
    void updateParam() {
        //pthread_join(dtid[brank], NULL);
        //LOG(INFO) << "diff from brother " << brank + 1 << " received";

        //displayParam();
        //loadParam(new_param.p);
        //displayParam();
        //setDiff(worker[brank].diff.p);
        //displayDiff();

        //LOG(INFO) << "parameters ready";

        //solver->Update();
        //net_->Update();
        //displayParam();

        // saveParam(new_param.p);             //把最新的参数保存下来
        
        LOG(INFO) << "started to save parameters\n";
        //把能更新的都更新一遍
        for (int i=0; i<task_count-1; i++) {

            if (pthread_mutex_trylock(m_pop + current_pop) != 0) {     //抢资源，存数据
                break;
            }
            setDiff(worker[current_pop].diff.p);
            pcenter->Update();
            pthread_join(dtid[current_pop], NULL);
            pthread_create(dtid + current_pop, NULL, waitForDiff, (void*)current_pop);//继续接收
            current_pop = (current_pop + 1) % (task_count-1);
        }
        
        pthread_join(sptid[frank], NULL);
        saveParam(worker[frank].param.p, pnet);   //并设为当前worker的待使用参数
        pthread_create(sptid + frank, NULL, sendParam, (void*)frank);
    }
    
    //以下代码用于做计算
    void compute() {
        pthread_join(rtid[frank], NULL);
        inputData(net_->bottom_vecs()[16][0]->data()->mutable_gpu_data(), net_->top_vecs()[0][1]->data()->mutable_gpu_data(), frank);

        loadParam(worker[frank].param.p, net_);

        float loss = 0.;
        loss = net_->ForwardFromTo(16,23);
        LOG(INFO) << "fc loss " << loss;
        net_->BackwardFromTo(23,16);
        LOG(INFO) << "fc has done for conv " << frank + 1;

        saveFcDiff(worker[frank].diff.p + conv_psz);    //保存fc diff
        outputData(net_->bottom_vecs()[16][0]->diff()->gpu_data(), frank); //保存残差


        pthread_join(srtid[frank], NULL);
        pthread_create(srtid + frank, NULL, sendResult, (void*)frank); //发送残差
        pthread_create(rtid + frank, NULL, waitForResult, (void*)frank);//重启结果接收   
    }
    

    void fc(int gpu) {
        device = gpu;

        init(); //分配空间及参数
         LOG(INFO) << "fc initialized";
        syncParams();   //所有机器同步参数
         LOG(INFO) << "fc parameter synchronized";
        startDiffReceiver();    //接收diff进程
        startResultReceiver();  //接收结果进程
         LOG(INFO) << "fc sub threads started";

        //net_->set_debug_info(23);
        LOG(INFO) << "dxy start all";

        //int ord[] = {0,2,4,1,3,5};
        //int bord[] = {1,3,5,0,2,4};
        int ord[] = {0,1};
        int bord[] = {1,0};
        //	solver->dxysave();
        for (int i=0; i<450000; i++) {	//最后调为450000
            printf("********  %d  *****\n", i);
            LOG(INFO) << "iter " << i << " started";
            solver->dxyite(i);
            if (i % 2000 == 0) solver->dxysave();
            frank = i % (task_count-1);     //这里比实际rank大小小1，0~5
            brank = bord[frank];             //兄弟
            frank = ord[frank];             //顺序要有些调整

            //net_->ClearParamDiffs();

            LOG(INFO) << "iter " << i << " initialized";
            compute();
            //displayLabel();
            LOG(INFO) << "bp for rank " << frank + 1 << " has done";

            updateParam();
            LOG(INFO) << "param updated";
        }
        LOG(INFO) << "dxy end backward";
        solver->dxysave();
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
};
