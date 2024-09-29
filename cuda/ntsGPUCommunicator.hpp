#ifndef NTSGPUCOMMUNICATION_HPP
#define NTSGPUCOMMUNICATION_HPP

#include "cuda_type.h"
#define CUDA_ENABLE 1
#if CUDA_ENABLE
#include "cuda_runtime.h"
#include <nccl.h>
#endif

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <thread>

//debug
#include <fstream>
#include <assert.h>
#include <unistd.h>//sleep

#include <mutex>
#include "ntsCUDA.hpp"

enum nccl_comm_type{sendmirror, sendsource, bcast};


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

const int DEFAULT_MESSAGEBUFFER_CAPACITY = 0;
typedef float ValueType;

inline size_t size_of_msg(int f_size) {
    // return sizeof(VertexId_CUDA) + sizeof(ValueType) * f_size;
    return sizeof(ValueType) * f_size;
}


// nn communicator
void initNCCLComm(ncclComm_t* comms, int nDev, int* devs);
void allReduceNCCL(void* send_buffer, void* recv_buffer, size_t element_num, ncclComm_t comm,
                   cudaStream_t cudaStream, int device_id);
void destroyNCCLComm(ncclComm_t comm);
void initNCCLRankComm(ncclComm_t* comms, int nDev, ncclUniqueId ncclId_, int device_id);

class NCCL_Communicator{
public:
    int device_num;
    int root;

    ncclComm_t* comm;

    NCCL_Communicator(int device_num_, int* devs, int root_=0)
    {
        this->root = root_;
        device_num = device_num_;
        comm = new ncclComm_t[device_num];
        initNCCLComm(comm, device_num, devs);
    }
    NCCL_Communicator(int device_num_, int root_=0)
    {
        this->root = root_;
        device_num = device_num_;
        comm = new ncclComm_t[device_num];
        ncclUniqueId ncclId;
        NCCLCHECK(ncclGetUniqueId(&ncclId));

        for(int i = 0; i < device_num; i++) {
            initNCCLRankComm(comm, device_num, ncclId, i);
        }
        
    }

    

    // ~NCCL_Communicator(){
    //     for(int i = 0; i < device_num; i++) {
    //         destroyNCCLComm(comm[i]);
    //     }
    //     delete []comm;
    // }
};


class NCCL_NN_Communicator{
public:
    NCCL_Communicator *nccl_comm;
    NCCL_NN_Communicator(NCCL_Communicator *nccl_comm_)
    {
        nccl_comm = nccl_comm_;
    }

    void AllReduce(int device_id, void* send_buffer, void* recv_buffer, size_t element_num,
                   cudaStream_t cudaStream) {
        allReduceNCCL(send_buffer, recv_buffer, element_num, nccl_comm->comm[device_id], cudaStream, device_id);
    }


};

struct GPU_MessageBuffer_Float{
    size_t capacity;
    int count;

    // char* data;
    ValueType* data;

    GPU_MessageBuffer_Float(Cuda_Stream *cs)
    {
        capacity = DEFAULT_MESSAGEBUFFER_CAPACITY;
        // count = (int *) cudaMallocGPU(sizeof(int));
        // CHECK_CUDA_RESULT(cudaMemset(count, 0, sizeof(int)))
        count = 0;
        // data = (char*) cudaMallocGPU(DEFAULT_MESSAGEBUFFER_CAPACITY, cs->stream);
        data = (ValueType*) cudaMallocGPU(DEFAULT_MESSAGEBUFFER_CAPACITY, cs->stream);
    }
    void resize_gpu_message_buffer(size_t new_capacity, Cuda_Stream *cs)
    {
        if(new_capacity < capacity)
        {
            return;
        }
        cudaFreeGPU(data, cs->stream);
        // data = (char*) cudaMallocGPU(new_capacity, cs->stream);
        data = (ValueType*) cudaMallocGPU(new_capacity, cs->stream);
        capacity = new_capacity;
    }
    void free_data(){
        if(data != nullptr)
        {
            cudaFree(data);
        }
    }
    ~GPU_MessageBuffer_Float(){
        if(data != nullptr)
        {
            cudaFree(data);
        }
    }
};


void nccl_point_to_point(GPU_MessageBuffer_Float **send_buffer, GPU_MessageBuffer_Float **recv_buffer, 
                         ncclComm_t comm, cudaStream_t cudaStream, int device_id, int device_num, int feature_size, 
                         volatile int &send_queue_size, int *send_queue, std::mutex &send_queue_mutex,
                         volatile int &recv_queue_size, int *recv_queue, std::mutex &recv_queue_mutex,
                         VertexId_CUDA *src_vtx, VertexId_CUDA *mir_vtx
                         );

void nccl_send_data_to_device(ValueType *send_data, int send_count, int trigger_device, ncclComm_t comm, cudaStream_t cudaStream);

void nccl_recv_data_to_device(ValueType *recv_data, int recv_count, int trigger_device, ncclComm_t comm, cudaStream_t cudaStream);

void nccl_broadcast(ValueType *send_data, ValueType *recv_data, int count, int root, ncclComm_t comm, cudaStream_t cudaStream);

class NCCL_graph_Communicator{
public:

    NCCL_Communicator* nccl_comm;
    Cuda_Stream *cs;

    VertexId_CUDA device_id;
    VertexId_CUDA *device_offset;
    VertexId_CUDA device_num;
    VertexId_CUDA current_send_device_id;

    nccl_comm_type nct;

    VertexId_CUDA feature_size;
    VertexId_CUDA owned_vertices;
    VertexId_CUDA *source_vertices;//对每个worker的
    VertexId_CUDA *mirror_vertices;

    GPU_MessageBuffer_Float **send_buffer;
    GPU_MessageBuffer_Float **recv_buffer;

    ValueType *f_input;
    ValueType *receive_input;

    int *send_queue;
    int *recv_queue;
    volatile int send_queue_size;
    volatile int recv_queue_size;

    // 现在还没实现到多线程控制通信
    std::mutex send_queue_mutex;
    std::mutex recv_queue_mutex;
    std::vector<std::thread> send_threads;
    std::vector<std::thread> recv_threads;
    std::thread *Send;
    std::thread *Recv;
    

    NCCL_graph_Communicator(VertexId_CUDA device_id_, VertexId_CUDA *device_offset_, VertexId_CUDA device_nums_,
                            VertexId_CUDA *source_vertices_, VertexId_CUDA *mirror_vertices_,
                            NCCL_Communicator* nccl_comm_, Cuda_Stream *cs_)
    {
        device_id = device_id_;
        device_offset = device_offset_;
        device_num = device_nums_;
        source_vertices = source_vertices_;
        mirror_vertices = mirror_vertices_;
        nccl_comm = nccl_comm_;
        cs = cs_;
        // cudaSetDevice(device_id);
        // initNCCLRankComm(&nccl_graph_comm, device_num, device_id);

        send_buffer = new GPU_MessageBuffer_Float*[device_num];
        recv_buffer = new GPU_MessageBuffer_Float*[device_num];

        for(int i = 0; i < device_num; i++)
        {
            send_buffer[i] = new GPU_MessageBuffer_Float(cs);
            recv_buffer[i] = new GPU_MessageBuffer_Float(cs);
        }

        
        // initNCCLRankComm(&nccl_comm, device_num, device_id);
    }
    
    void release_nccl_graph_comm()
    {
        delete[] send_queue;
        delete[] recv_queue;
        // for(int i = 0; i < device_num; i++)
        // {   
        //     if(i != device_id)
        //     {
        //         send_buffer[i]->free_data();
        //         recv_buffer[i]->free_data();
        //     }
        // }
    }

    //full graph的发给本地的不需要buffer，所以这里初始化buffer不带本地
    void init_nccl_layer_all_full_graph(int feature_size_, nccl_comm_type nct_)
    {
        send_queue = new int[device_num];
        recv_queue = new int[device_num];
        send_queue_size = 0;
        recv_queue_size = 0;
        feature_size = feature_size_;

        nct = nct_;

        //master2mirror
        if(nct == sendmirror){
            for(int i = 0; i < device_num; i++)
            {
                if(i != device_id)
                {
                    send_buffer[i]->resize_gpu_message_buffer(mirror_vertices[i] * size_of_msg(feature_size), cs);
                    recv_buffer[i]->resize_gpu_message_buffer(source_vertices[i] * size_of_msg(feature_size), cs);
                }
                send_buffer[i]->count = 0;
                recv_buffer[i]->count = 0;
            }
        } else if(nct == sendsource){
            for(int i = 0; i < device_num; i++)
            {
                if(i != device_id)
                {
                    send_buffer[i]->resize_gpu_message_buffer(source_vertices[i] * size_of_msg(feature_size), cs);
                    recv_buffer[i]->resize_gpu_message_buffer(mirror_vertices[i] * size_of_msg(feature_size), cs);
                }
                send_buffer[i]->count = 0;
                recv_buffer[i]->count = 0;
            }
        }
    }

    void set_current_send_device(VertexId_CUDA csdi)
    {
        current_send_device_id = csdi;
    }

    //if minibatch, we will send id + embedding, so the type of buffer->data should be char
    void emit_buffer(ValueType * f_input, VertexId_CUDA *vtx, VertexId_CUDA index)
    {
        int pos = __sync_fetch_and_add(&send_buffer[current_send_device_id]->count, 1);

        CHECK_CUDA_RESULT(cudaMemcpy(send_buffer[current_send_device_id]->data+index*size_of_msg(feature_size), 
                                    vtx, sizeof(VertexId_CUDA), cudaMemcpyDeviceToDevice));

        CHECK_CUDA_RESULT(cudaMemcpy(send_buffer[current_send_device_id]->data+index*size_of_msg(feature_size) + sizeof(VertexId_CUDA), 
                                    f_input, sizeof(float)*feature_size, cudaMemcpyDeviceToDevice));                      
    }

    //when we have the message index, we put the buffer rank by index, wo can only send embedding without id
    void emit_buffer_only_embedding(ValueType * f_input, VertexId_CUDA index)
    {
        int pos = __sync_fetch_and_add(&send_buffer[current_send_device_id]->count, 1);

        CHECK_CUDA_RESULT(cudaMemcpy(send_buffer[current_send_device_id]->data + index * feature_size, 
                                    f_input, sizeof(ValueType)*feature_size, cudaMemcpyDeviceToDevice));                      
    }

    //when we have the message index, we put the buffer rank by index, wo can only send embedding without id
    void emit_buffer_only_embedding(ValueType * f_input, VertexId_CUDA index, VertexId_CUDA vtx_num)
    {
        int pos = __sync_fetch_and_add(&send_buffer[current_send_device_id]->count, vtx_num);

        CHECK_CUDA_RESULT(cudaMemcpy(send_buffer[current_send_device_id]->data + index * feature_size, 
                                    f_input, sizeof(ValueType)*feature_size*vtx_num, cudaMemcpyDeviceToDevice));                      
    }

    void store_input_full_graph(ValueType *f_input_)
    {
        f_input = f_input_;
    }

    void trigger_one_device(VertexId_CUDA trigger_device)
    {
        if(nct == sendmirror){
            recv_buffer[trigger_device]->count = source_vertices[trigger_device];
        }
        else if(nct == sendsource){
            recv_buffer[trigger_device]->count = mirror_vertices[trigger_device];
        }
        if(device_id == trigger_device)
        {
            device_is_ready_for_recv(trigger_device);
        }
        else
        {
            device_is_ready_for_send(trigger_device);
        }
    }

    void device_is_ready_for_recv(VertexId_CUDA trigger_device)
    {
        recv_queue[recv_queue_size] = trigger_device;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
    }

    void device_is_ready_for_send(VertexId_CUDA trigger_device) {
        if (device_id!= trigger_device) {
            send_queue[send_queue_size] = trigger_device;
            send_queue_mutex.lock();
            send_queue_size += 1;
            send_queue_mutex.unlock();
        }
    }

    void point_to_point()
    {
        nccl_point_to_point(send_buffer, recv_buffer, nccl_comm->comm[device_id], 
                            cs->stream, device_id, device_num, feature_size,
                            send_queue_size, send_queue, send_queue_mutex, 
                            recv_queue_size, recv_queue, recv_queue_mutex,
                            source_vertices, mirror_vertices);

        // cs->CUDA_DEVICE_SYNCHRONIZE();
    }

    void send_data()
    {
        std::ofstream out("./log/cora_comm_" + std::to_string(device_id) + ".txt", std::ios_base::out);//for debug
        for(int step = 0; step < device_num; step++)
        {
            if(step == device_num - 1)
            {
                break;
            }

            // while(true)//这个语法是一直检测send queue的内容，后期优化的时候加上这块
            // {
            //     send_queue_mutex.lock();
            //     bool condition = (send_queue_size <= step);
            //     send_queue_mutex.unlock();
            //     if(!condition)
            //     {
            //         break;
            //     }
            // }

            int trigger_device = send_queue[step];
            out << "trigger device:" << trigger_device << std::endl;

            

        }
    }

    void recv_data()
    {

    }

    ValueType *receive_one_device_full_graph(VertexId_CUDA &trigger_device, int step)
    {//由于现在这个版本的代码中，到这里数据已经发送完毕，所以无需while(true)等待。后续优化加上这里！
        // while (true) {
        //     recv_queue_mutex.lock();
        //     bool condition = (recv_queue_size <= step);
        //     recv_queue_mutex.unlock();
        //     if (!condition)
        //         break;
        //     __asm volatile("pause" ::: "memory");
        // }

        int i = recv_queue[step];
        trigger_device = i;

        if(i == device_id){
            return f_input;
        }
        else{
            return recv_buffer[i]->data;
        }
    }
    
    ValueType *buffer2mirror(VertexId_CUDA trigger_device)
    {
        std::cout << "device:" << device_id << " trigger device:" << trigger_device << "count" 
                    << recv_buffer[trigger_device]->count << std::endl;
        ValueType *mirror = (ValueType *)cudaMallocGPU(sizeof(ValueType) * feature_size * recv_buffer[trigger_device]->count, cs->stream);
        
        
        return mirror;
    }
    
    void broadcast(ValueType* send_data, ValueType* recv_data,VertexId_CUDA feature_size_, VertexId_CUDA vtx_count, int root)
    {
        feature_size = feature_size_;
        nct = bcast;
        nccl_broadcast(send_data, recv_data, feature_size*vtx_count, root, nccl_comm->comm[device_id], cs->stream);
    }

    void debug_broadcast(ValueType* recv_data, int count ,int step)//count:程序被调用的次数
    {
        std::ofstream out("./log/debug_broadcast" + std::to_string(device_id) + ".txt", std::ios_base::out);
        out << "------------------------broadcast step: " << step << std::endl;
        out << "feature size: " << feature_size << std::endl;
        cudaPointerAttributes attributes;
        CHECK_CUDA_RESULT(cudaPointerGetAttributes(&attributes, recv_data));
        out << "(send buffer)GPU ID: " << attributes.device << std::endl;
        ValueType * data = (ValueType*)malloc(size_of_msg(feature_size) * count);
        CHECK_CUDA_RESULT(cudaMemcpy(data, recv_data, 
                                        size_of_msg(feature_size) * count, cudaMemcpyDeviceToHost));
        
        for(int i = 0; i < count; i++)
        {
            out << "vtx index: " << i  << "   feat: ";
            for(int j = 0; j < feature_size; j++)
            {
                ValueType d = data[i * feature_size + j];
                out << j << ":" <<  d << " ";
            }
            out << std::endl;
        }
    }
    

    void comm_only_embedding_debug()
    {
        std::ofstream out("./log/cora_comm_only_embedding_" + std::to_string(device_id) + ".txt", std::ios_base::out);
        out << "send_queue_size:" << send_queue_size << std::endl;
        for(VertexId_CUDA trigger_device = 0; trigger_device < device_num; trigger_device++)
        {
            out << "------------------------send device: " << trigger_device << std::endl;
            out << "------------------------send count: " << send_buffer[trigger_device]->count << std::endl;
            out << "feature size: " << feature_size << std::endl;
            cudaPointerAttributes attributes;
            CHECK_CUDA_RESULT(cudaPointerGetAttributes(&attributes, send_buffer[trigger_device]->data));
            out << "(send buffer)GPU ID: " << attributes.device << std::endl;
            ValueType * send_data = (ValueType*)malloc(size_of_msg(feature_size) * send_buffer[trigger_device]->count);
            CHECK_CUDA_RESULT(cudaMemcpy(send_data, send_buffer[trigger_device]->data, 
                                          size_of_msg(feature_size) * send_buffer[trigger_device]->count, cudaMemcpyDeviceToHost));
            
            for(int i = 0; i < send_buffer[trigger_device]->count; i++)
            {
              out << "vtx index: " << i  << "   feat: ";
              for(int j = 0; j < feature_size; j++)
              {
                ValueType data = send_data[i * feature_size + j];
                out << j << ":" <<  data << " ";
              }
              out << std::endl;
            }
        }

        for(VertexId_CUDA step = 0; step < device_num; step++)
        {
             //print recv buffer
            int trigger_device = recv_queue[step];
            out << "-----------------------recv device: " << trigger_device << std::endl;
            out << "-----------------------recv count: " << recv_buffer[trigger_device]->count << std::endl;
            out << "-----------------------recv queue[" << step << "]: " << recv_queue[step]  << std::endl;
            out << "feature size: " << feature_size << std::endl;
            
            ValueType *f_inpui_data = (ValueType*)malloc(sizeof(ValueType) * feature_size * recv_buffer[trigger_device]->count);

            if(trigger_device == device_id)
            {
                cudaPointerAttributes attributes_;
                CHECK_CUDA_RESULT(cudaPointerGetAttributes(&attributes_,f_input));
                out << "(receice buffer)GPU ID: " << attributes_.device << std::endl;
                CHECK_CUDA_RESULT(cudaMemcpy(f_inpui_data, f_input, 
                                    sizeof(ValueType) * feature_size * recv_buffer[trigger_device]->count, cudaMemcpyDeviceToHost));
            
            }
            else{
                cudaPointerAttributes attributes;
                CHECK_CUDA_RESULT(cudaPointerGetAttributes(&attributes, recv_buffer[trigger_device]->data));
                out << "(receice buffer)GPU ID: " << attributes.device << std::endl;
                CHECK_CUDA_RESULT(cudaMemcpy(f_inpui_data, recv_buffer[trigger_device]->data, 
                                        size_of_msg(feature_size) * recv_buffer[trigger_device]->count, cudaMemcpyDeviceToHost));
                out << "finish copy recv data to cpu" << std::endl;
            }

            for(int i = 0; i < recv_buffer[trigger_device]->count; i++)
            {
                out << "v index: " << i << "   feat: ";
                for(int j = 0; j < feature_size; j++)
                {
                    out << j << ":" << f_inpui_data[i * feature_size + j] << " ";
                }
                out << std::endl;
            }
        }
        out << "recv queue size : " << recv_queue_size << std::endl;
        out << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!finish debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        // sleep(5);
        
    }






};


class NCCL_PT_Communicator{
public:
    NCCL_Communicator* nccl_comm;
    Cuda_Stream *cs;

    VertexId_CUDA device_id;//global device id
    VertexId_CUDA global_device_num;
    VertexId_CUDA P_num;
    
    VertexId_CUDA feature_size;

    // GPU_MessageBuffer_Float **send_buffer;
    GPU_MessageBuffer_Float **recv_buffer;

    VertexId_CUDA *msg_offset;

    // int *send_queue;
    int *recv_queue;
    // volatile int send_queue_size;
    volatile int recv_queue_size;

    NCCL_PT_Communicator(VertexId_CUDA _device_id, VertexId_CUDA global_device_num_, VertexId_CUDA P_num_,
                            NCCL_Communicator* _nccl_comm, Cuda_Stream * _cs, VertexId_CUDA *msg_offset_)
    {
        device_id = _device_id;
        global_device_num = global_device_num_;
        P_num = P_num_;
        nccl_comm = _nccl_comm;
        cs = _cs;

        msg_offset = msg_offset_;

    }

    void release_nccl_pt_comm()
    {
        // delete[] send_queue;
        delete[] recv_queue;
        // for(int i = 0; i < device_num; i++)
        // {   
        //     if(i != device_id)
        //     {
        //         send_buffer[i]->free_data();
        //         recv_buffer[i]->free_data();
        //     }
        // }
    }
    
    // void init_nccl_send_to_P(int feature_size_)
    // {
    //     feature_size = feature_size_;
    //     // send_queue = new int[msg_device_num];
    //     // send_queue_size = 0;
    //     // printf("send root : %d\n", msg_root);
    //     // printf("msg_device_num : %d\n", msg_device_num);
    //     // printf("msg feature size : %d\n", feature_size);
    //     // send_buffer = new GPU_MessageBuffer_Float*[msg_device_num];
    //     // for(int i = 0; i < msg_device_num; i++)
    //     // {
    //         // send_buffer[i] = new GPU_MessageBuffer_Float(cs);
    //         // send_buffer[i]->resize_gpu_message_buffer(
    //         //         (msg_offset[msg_root + i + 1] - msg_offset[msg_root + i]) * size_of_msg(feature_size), cs);
    //         // send_buffer[i]->count = 0;            
    //         // printf("device[%d] send to device[%d] vertices num : %d\n", device_id, msg_root + i,
    //         //                                                             msg_offset[msg_root + i + 1] - msg_offset[msg_root + i]);
    //     // }
    // }

    void T2P_send_from_X_buffer(int feature_size_, ValueType *f_input)
    {
        NCCLCHECK(ncclGroupStart());
        feature_size = feature_size_;
        
        int index = 0;
        for(int step = 0; step < P_num; step++)
        {
            int trigger_device = step;//P device是从0开始
            int send_vtx_num = msg_offset[trigger_device + 1] - msg_offset[trigger_device];
            if(send_vtx_num == 0)
            {
                continue;
            }
            // nccl_send_data_to_device(f_input + index * feature_size, send_vtx_num * size_of_msg(feature_size), 
            nccl_send_data_to_device(f_input + index * feature_size, send_vtx_num * feature_size, 
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);
            
            ValueType * recv = (ValueType *) cudaMallocGPU(sizeof(ValueType) * 1, cs->stream);
            nccl_recv_data_to_device(recv, 1, //发了不收就不让啊
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);

            index += send_vtx_num;
            
            // printf("device[%d] send to device[%d] vertices num : %d\n", device_id, trigger_device, send_vtx_num);
        }
        
        NCCLCHECK(ncclGroupEnd());
    }

    void T2P_recv_write_to_X_buffer(int feature_size_, ValueType *f_output)
    {
        NCCLCHECK(ncclGroupStart());
        feature_size = feature_size_;

        int index = 0;
        for(int step = 0; step < global_device_num - P_num; step++)
        {
            int trigger_device = step + P_num;
            int recv_vtx_num = msg_offset[step + 1] - msg_offset[step];
            if(recv_vtx_num == 0)
            {
                continue;
            }

            ValueType * send = (ValueType *) cudaMallocGPU(sizeof(ValueType) * 1, cs->stream);
            CHECK_CUDA_RESULT(cudaMemset(send, 1.0, sizeof(ValueType) * 1));
            nccl_send_data_to_device(send, 1, 
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);

            // nccl_recv_data_to_device(f_output + index * feature_size, recv_vtx_num * size_of_msg(feature_size), 
            nccl_recv_data_to_device(f_output + index * feature_size, recv_vtx_num * feature_size, 
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);
            index += recv_vtx_num;
            
            // printf("device[%d] recv from device[%d] vertices num : %d\n", device_id, trigger_device, recv_vtx_num);
        }
        
        NCCLCHECK(ncclGroupEnd());
    }

    void P2T_send_from_X_buffer(int feature_size_, ValueType *f_input)
    {
        NCCLCHECK(ncclGroupStart());
        feature_size = feature_size_;
        
        int index = 0;
        for(int step = 0; step < global_device_num -  P_num; step++)
        {
            int trigger_device = step + P_num;//P device是从0开始
            int send_vtx_num = msg_offset[step + 1] - msg_offset[step];
            if(send_vtx_num == 0)
            {
                continue;
            }
            nccl_send_data_to_device(f_input + index * feature_size, send_vtx_num * feature_size, 
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);
            
            ValueType * recv = (ValueType *) cudaMallocGPU(sizeof(ValueType) * 1, cs->stream);
            nccl_recv_data_to_device(recv, 1, //发了不收就不让啊
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);

            index += send_vtx_num;
        }
        
        NCCLCHECK(ncclGroupEnd());
    }

    void P2T_recv_write_to_X_buffer(int feature_size_, ValueType *f_output)
    {
        NCCLCHECK(ncclGroupStart());
        feature_size = feature_size_;

        int index = 0;
        for(int step = 0; step < P_num; step++)
        {
            int trigger_device = step;
            int recv_vtx_num = msg_offset[step + 1] - msg_offset[step];
            if(recv_vtx_num == 0)
            {
                continue;
            }

            ValueType * send = (ValueType *) cudaMallocGPU(sizeof(ValueType) * 1, cs->stream);
            CHECK_CUDA_RESULT(cudaMemset(send, 1.0, sizeof(ValueType) * 1));
            nccl_send_data_to_device(send, 1, 
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);

            // nccl_recv_data_to_device(f_output + index * feature_size, recv_vtx_num * size_of_msg(feature_size), 
            nccl_recv_data_to_device(f_output + index * feature_size, recv_vtx_num * feature_size, 
                                    trigger_device, nccl_comm->comm[device_id], cs->stream);
            index += recv_vtx_num;
            
            // printf("device[%d] recv from device[%d] vertices num : %d\n", device_id, trigger_device, recv_vtx_num);
        }
        
        NCCLCHECK(ncclGroupEnd());
    }


};


#endif