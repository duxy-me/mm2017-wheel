LIBRARIES=glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5 lmdb opencv_core opencv_highgui opencv_imgproc  boost_thread stdc++  cudnn cublas cudart caffe pthread
CAFFE_ROOT=/home/dxy/software/caffe/caffe-master

mpi_final:
	mpic++ main_final.cpp -I ${CAFFE_ROOT}/include/ -I /usr/local/cuda/include -I ${CAFFE_ROOT}/.build_release/src/ -Wl,-rpath,${CAFFE_ROOT}/build/lib -L${CAFFE_ROOT}/build/lib -L /usr/local/cuda/lib64/ $(foreach library,$(LIBRARIES),-l$(library))  -o main


caffe:
	 g++ caffe.cpp -pthread -fPIC -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -O2 -DUSE_CUDNN -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy-1.11.0-py2.7-linux-x86_64.egg/numpy/core/include -I/usr/local/include -I${CAFFE_ROOT}/.build_release/src -I${CAFFE_ROOT}/src -I${CAFFE_ROOT}/include -I/usr/local/cuda/include -Wall -Wno-sign-compare -lcaffe -L/usr/lib -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L${CAFFE_ROOT}/.build_release/lib -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lboost_thread -lstdc++ -lcudnn -lcblas -latlas -Wl,-rpath,${CAFFE_ROOT}/build/lib -o main

clean:
	rm main
