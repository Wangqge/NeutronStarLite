
#include <random>
#include"cuda_type.h"
#include "ntsGPUCommunicator.hpp"







void initNCCLComm(ncclComm_t* comms, int nDev, int* devs) {
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
}

void destroyNCCLComm(ncclComm_t comm) {
    NCCLCHECK(ncclCommDestroy(comm));
}


void allReduceNCCL(void* send_buffer, void* recv_buffer, size_t element_num, ncclComm_t comm,
                   cudaStream_t cudaStream, int device_id) {
    CHECK_CUDA_RESULT(cudaSetDevice(device_id));
    NCCLCHECK(ncclAllReduce(send_buffer, recv_buffer, element_num, ncclFloat,
                            ncclSum, comm, cudaStream));
}


//暂时没用到
void initNCCLRankComm(ncclComm_t* comms, int nDev, ncclUniqueId ncclId_, int device_id)
{
    // NCCLCHECK(ncclGroupStart());
    std::cout << "init dev comm : " << device_id <<std::endl;
    NCCLCHECK(ncclCommInitRank(&comms[device_id], nDev, ncclId_, device_id));
    // NCCLCHECK(ncclGroupEnd());
}


//ncclSend/ncclRecv is not block when you give the cudaStream. 
//if we set cudaStream=0(default), it will be blocked, so we don't need to set cudaStreamSynchronize.
void nccl_point_to_point(GPU_MessageBuffer_Float **send_buffer, GPU_MessageBuffer_Float **recv_buffer, 
                         ncclComm_t comm, cudaStream_t cudaStream, int device_id, int device_num, int feature_size, 
                         volatile int &send_queue_size, int *send_queue, std::mutex &send_queue_mutex,
                         volatile int &recv_queue_size, int *recv_queue, std::mutex &recv_queue_mutex,
                         VertexId_CUDA *src_vtx, VertexId_CUDA *mir_vtx
                         )
{
    NCCLCHECK(ncclGroupStart());
    for(int step = 0; step < device_num; step++)
    {
      if(step == device_num - 1)
      {
        break;
      }

      int trigger_device = send_queue[step];

      assert(device_id != trigger_device);
      
      // NCCLCHECK(ncclSend(send_buffer[trigger_device]->data, size_of_msg(feature_size) * send_buffer[trigger_device]->count,
      //                     ncclChar, trigger_device, comm, 0));
      
      //send_buffer[trigger_device]->count 是发送的时候算的
      // NCCLCHECK(ncclSend(send_buffer[trigger_device]->data, size_of_msg(feature_size) * send_buffer[trigger_device]->count,
      //                     ncclFloat, trigger_device, comm, 0));
      NCCLCHECK(ncclSend(send_buffer[trigger_device]->data, feature_size * send_buffer[trigger_device]->count,
                          ncclFloat, trigger_device, comm, 0));
    }

    for(int step = 1; step < device_num; step++)
    {
      int trigger_device = (device_id + step) % device_num;

      //怎么确定recv的count呢？
      //其实就是device_id->trigger_device的source点的数量
      
      // NCCLCHECK(ncclRecv(recv_buffer[trigger_device]->data, size_of_msg(feature_size) * recv_buffer[trigger_device]->count,
      //                     ncclChar, trigger_device, comm, 0));

      //recv_buffer[trigger_device]->count 是在trigger_one_device的时候根据nccl_comm_type设置的
      // NCCLCHECK(ncclRecv(recv_buffer[trigger_device]->data, size_of_msg(feature_size) * recv_buffer[trigger_device]->count,
      //                     ncclFloat, trigger_device, comm, 0));
      NCCLCHECK(ncclRecv(recv_buffer[trigger_device]->data, feature_size * recv_buffer[trigger_device]->count,
                          ncclFloat, trigger_device, comm, 0));
      // out << "--------------finish recv!!!!" << std::endl;
      
    }
    NCCLCHECK(ncclGroupEnd());
    // CHECK_CUDA_RESULT(cudaStreamSynchronize(cudaStream));

    for(int step = 1; step < device_num; step++)
    {
      int trigger_device = (device_id + step) % device_num;
      
      recv_queue[recv_queue_size] = trigger_device;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();

    }


}

void nccl_broadcast(ValueType *send_data, ValueType *recv_data, int count, int root, ncclComm_t comm, cudaStream_t cudaStream)
{
  NCCLCHECK(ncclBcast(send_data, count, ncclFloat, root, comm, cudaStream));
  // NCCLCHECK(ncclBroadcast(send_data, recv_data, count, ncclFloat, root, comm, cudaStream));
}

void nccl_send_data_to_device(ValueType *send_data, int send_count, int trigger_device, ncclComm_t comm, cudaStream_t cudaStream)
{
    NCCLCHECK(ncclSend(send_data, send_count, ncclFloat, trigger_device, comm, cudaStream));
    // NCCLCHECK(ncclSend(send_data, send_count, ncclFloat, trigger_device, comm, 0));
}

void nccl_recv_data_to_device(ValueType *recv_data, int recv_count, int trigger_device, ncclComm_t comm, cudaStream_t cudaStream)
{
    NCCLCHECK(ncclRecv(recv_data, recv_count, ncclFloat, trigger_device, comm, cudaStream));
    // NCCLCHECK(ncclRecv(recv_data, recv_count, ncclFloat, trigger_device, comm, 0));
}