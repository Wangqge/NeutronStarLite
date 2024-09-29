
#ifndef DEVICESUBSTRUCTURE_HPP
#define DEVICESUBSTRUCTURE_HPP
#include <vector>
#include <set>

#include<fstream>


#include "core/FullyRepGraph.hpp"
#include "core/graph.hpp"
#include "core/ntsDataloador.hpp"
#include "core/NtsScheduler.hpp"
#include <c10/cuda/CUDAStream.h>


class SubgraphSegment
{
public:

    

    VertexId *column_offset; // VertexNumber
    VertexId *row_indices;   // edge_size also the source nodes,   indices 都是global id
    VertexId *row_offset;    // VertexNumber
    VertexId *column_indices;//global id

    std::vector<VertexId> source_vertices;//去重之后的source点,全局id
    std::vector<VertexId> mirror_vertices;//全局id

    long *source;
    long *destination;

    ValueType *edge_weight_forward; // edge_size
    ValueType *edge_weight_backward;

    VertexId *forward_message_index;
    VertexId *backward_message_index;

    VertexId *column_offset_gpu; // VertexNumber
    VertexId *row_indices_gpu;
    VertexId *row_offset_gpu;     // VertexNumber
    VertexId *column_indices_gpu; // edge_size

    VertexId* mirror_vertices_gpu;

    ValueType *edge_weight_forward_gpu;  // edge_size
    ValueType *edge_weight_backward_gpu; // edge_size

    VertexId *forward_message_index_gpu; // edge_size
    VertexId *backward_message_index_gpu;

    //init初始化的
    int edge_size;
    int batch_size_forward;
    int batch_size_backward;
    int feature_size;
    int src_range[2];
    int dst_range[2];

    int input_size;
    int output_size;

    bool is_CPU;

    ~SubgraphSegment()
    {
        delete[] forward_message_index;
        delete[] backward_message_index;

    }
    
    void init(VertexId src_start, VertexId src_end,
                VertexId dst_start, VertexId dst_end,
                VertexId edge_size_, bool is_CPU_ = false) {
        src_range[0] = src_start;
        src_range[1] = src_end;
        dst_range[0] = dst_start;
        dst_range[1] = dst_end;
        batch_size_backward = src_range[1] - src_range[0];
        batch_size_forward = dst_range[1] - dst_range[0];//本地顶点的数量
        edge_size = edge_size_;
        is_CPU = is_CPU_;
    }

    void allocVertexAssociateData()
    {//init offset
        column_offset =
            (VertexId *)malloc((batch_size_forward + 1) * sizeof(VertexId));
        row_offset =
            (VertexId *)malloc((batch_size_backward + 1) * sizeof(VertexId));
        memset(column_offset, 0, (batch_size_forward + 1) * sizeof(VertexId));
        memset(row_offset, 0, (batch_size_backward + 1) * sizeof(VertexId));
    }

    void allocEdgeAssociateData()
    {//nts源码为啥是edge size + 1 呢？
        row_indices = (VertexId *)malloc((edge_size ) * sizeof(VertexId));
        memset(row_indices, 0, (edge_size) * sizeof(VertexId));
        edge_weight_forward =
            (ValueType *)malloc((edge_size) * sizeof(ValueType));
        memset(edge_weight_forward, 0, (edge_size) * sizeof(VertexId));
        column_indices = (VertexId *)malloc((edge_size) * sizeof(VertexId)); ///
        memset(column_indices, 0, (edge_size) * sizeof(VertexId));
        edge_weight_backward =
            (ValueType *)malloc((edge_size) * sizeof(ValueType)); ///
        memset(edge_weight_backward, 0, (edge_size) * sizeof(ValueType));
        destination = (long *)malloc((edge_size) * sizeof(long));
        memset(destination, 0, (edge_size) * sizeof(long));
        source = (long *)malloc((edge_size) * sizeof(long));
        memset(source, 0, (edge_size) * sizeof(long));
    }

    void generate_message_index()
    {
        forward_message_index = new VertexId[batch_size_backward];
        backward_message_index = new VertexId[mirror_vertices.size()];

        int index = 0;
        for(int i = 0; i < source_vertices.size(); i++)
        {
            assert((source_vertices[i]-src_range[0]) < batch_size_backward);   
            forward_message_index[source_vertices[i] - src_range[0]] = index++;
        }

        for(int i = 0; i < mirror_vertices.size(); i++)
        {
            assert((mirror_vertices[i]-dst_range[0]) < batch_size_forward);   
            backward_message_index[i] = mirror_vertices[i] - dst_range[0];
        }
    }

    void load_graph_to_GPUs(cudaStream_t cuda_stream)
    {
        column_offset_gpu = (VertexId *)cudaMallocGPU((batch_size_forward + 1) * sizeof(VertexId), cuda_stream);
        row_offset_gpu = (VertexId *)cudaMallocGPU((batch_size_backward + 1) * sizeof(VertexId), cuda_stream);
        column_indices_gpu = (VertexId *)cudaMallocGPU(edge_size * sizeof(VertexId), cuda_stream);
        row_indices_gpu = (VertexId *)cudaMallocGPU(edge_size * sizeof(VertexId), cuda_stream);
        edge_weight_forward_gpu = (ValueType *)cudaMallocGPU(edge_size * sizeof(ValueType), cuda_stream);
        edge_weight_backward_gpu = (ValueType *)cudaMallocGPU(edge_size * sizeof(ValueType), cuda_stream);//这个不一定需要，因为只是跟上一个顺序不一样
        // mirror_vertices_gpu = (VertexId *)cudaMallocGPU(mirror_vertices.size() * sizeof(VertexId), cuda_stream);

        forward_message_index_gpu = (VertexId *)cudaMallocGPU(batch_size_backward * sizeof(VertexId), cuda_stream);
        backward_message_index_gpu = (VertexId *)cudaMallocGPU(mirror_vertices.size() * sizeof(VertexId), cuda_stream);

        move_bytes_in(column_offset_gpu, column_offset, (batch_size_forward + 1) * sizeof(VertexId));
        move_bytes_in(row_offset_gpu, row_offset, (batch_size_backward + 1) * sizeof(VertexId));
        move_bytes_in(column_indices_gpu, column_indices, edge_size * sizeof(VertexId));
        move_bytes_in(row_indices_gpu, row_indices, edge_size * sizeof(VertexId));
        move_bytes_in(edge_weight_forward_gpu, edge_weight_forward, edge_size * sizeof(ValueType));
        move_bytes_in(edge_weight_backward_gpu, edge_weight_backward, edge_size * sizeof(ValueType));
        // move_bytes_in(mirror_vertices_gpu, mirror_vertices.data(), mirror_vertices.size() * sizeof(VertexId));

        
        move_bytes_in(forward_message_index_gpu, forward_message_index, batch_size_backward * sizeof(VertexId));
        move_bytes_in(backward_message_index_gpu, backward_message_index, mirror_vertices.size() * sizeof(VertexId));


    }

    void load_graph_tp_per_gpu(cudaStream_t cuda_stream)
    {
        column_offset_gpu = (VertexId *)cudaMallocGPU((batch_size_forward + 1) * sizeof(VertexId), cuda_stream);
        row_offset_gpu = (VertexId *)cudaMallocGPU((batch_size_backward + 1) * sizeof(VertexId), cuda_stream);
        column_indices_gpu = (VertexId *)cudaMallocGPU(edge_size * sizeof(VertexId), cuda_stream);
        row_indices_gpu = (VertexId *)cudaMallocGPU(edge_size * sizeof(VertexId), cuda_stream);
        edge_weight_forward_gpu = (ValueType *)cudaMallocGPU(edge_size * sizeof(ValueType), cuda_stream);
        edge_weight_backward_gpu = (ValueType *)cudaMallocGPU(edge_size * sizeof(ValueType), cuda_stream);

        move_bytes_in(column_offset_gpu, column_offset, (batch_size_forward + 1) * sizeof(VertexId));
        move_bytes_in(row_offset_gpu, row_offset, (batch_size_backward + 1) * sizeof(VertexId));
        move_bytes_in(column_indices_gpu, column_indices, edge_size * sizeof(VertexId));
        move_bytes_in(row_indices_gpu, row_indices, edge_size * sizeof(VertexId));
        move_bytes_in(edge_weight_forward_gpu, edge_weight_forward, edge_size * sizeof(ValueType));
        move_bytes_in(edge_weight_backward_gpu, edge_weight_backward, edge_size * sizeof(ValueType));
    }

    void free_graph_from_per_gpu(cudaStream_t cuda_stream){
        cudaFreeGPU(column_offset_gpu,cuda_stream);
        cudaFreeGPU(row_offset_gpu,cuda_stream);
        cudaFreeGPU(column_indices_gpu,cuda_stream);
        cudaFreeGPU(row_indices_gpu,cuda_stream);
        cudaFreeGPU(edge_weight_forward_gpu,cuda_stream);
        cudaFreeGPU(edge_weight_backward_gpu,cuda_stream);
    }


};



class DeviceSubStructure
{
private:
    
public:
    int max_threads;//设备可用的总线程数量

    Cuda_Stream *cs;
    at::cuda::CUDAStream *ts;
    //local topo: 全局id
    VertexId *dstList;
    VertexId *srcList;
    VertexId *source_vtx;// source 点的全局index，for GAT

    //meta info
    Graph<Empty> *graph;
    int root;
    VertexId all_device_num;
    VertexId P_num;
    VertexId* subdevice_offset;
    VertexId device_num;
    VertexId device_id;//local id (device_id+root为真正的GPU id)
    VertexId global_device_id;//
    VertexId global_vertices;
    VertexId global_edges;

    VertexId owned_vertices;
    VertexId owned_edges;
    VertexId owned_sources;//source 点的数量
    
    //graph with only local dst vertices;
    //其实暂时没有用这个，尽管是partitioned graph那个类也没有使用这个变量,这个是compress的
    VertexId* column_offset;
    VertexId* row_indices;
    VertexId* row_offset;
    VertexId* column_indices;

    //graph segment
    std::vector<SubgraphSegment*> graph_chunks;

    //communication
    std::vector<VertexId*> mirror_of_device; //[device_id][id]
    VertexId** mirror_of_device_gpu;

    // NN information
    std::vector<NtsVar> X;
    std::vector<NtsVar> X_P;
    NtsVar decoupled_mid_grad;
    NtsVar feature;
    NtsVar label;
    NtsVar label_gpu;
    NtsVar mask;
    NtsVar mask_gpu;
    std::vector<Parameter*> P;
    int feature_size;
    VertexId * P2T_offset;//size：num of T device + 1，offset[id + 1] - offset[id] 为 当前device应该发改id+root device的顶点数量
    VertexId * T2P_offset;//这个与上一个相反。  即local id：offset[id] ~~ offset[id + 1] 的点，embedding发送给第id+root个device

    NCCL_graph_Communicator *nccl_graph_comm;

    NCCL_PT_Communicator *nccl_pt_comm;

    bool is_CPU = false;

    DeviceSubStructure(Graph<Empty> *graph_, VertexId global_v_, VertexId global_e_,
                        VertexId* subdevice_offset_, VertexId device_num_, VertexId device_id_,
                        VertexId owned_edges_, int feature_size_, int root_ = 0, int layer_deep = 0)
    {
        graph = graph_;
        root = root_;
        subdevice_offset = subdevice_offset_;
        device_num = device_num_;
        device_id = device_id_;
        global_vertices = global_v_;
        global_edges = global_e_;
        owned_vertices = subdevice_offset[device_id+1]-subdevice_offset[device_id];
        owned_edges = owned_edges_;
        feature_size = feature_size_;

        global_device_id = device_id + root;

        // max_threads = omp_get_thread_num();
        // std::cout <<"244 max threads:  " << max_threads << std::endl;

        if(layer_deep == 0){
            layer_deep = graph->gnnctx->layer_size.size();
        }
        printf("device[%d]layer_deep : %d\n", global_device_id, layer_deep - 1);
        for (int i = 0; i < layer_deep; i++) {
            NtsVar d;
            X.push_back(d);
        }

        if(device_num == 0)
        {
            device_num = 1;
            is_CPU = true;
        }
        else
        {
            is_CPU = false;
        }

        
        if(!is_CPU)
        {     
            init_CudaStream();
        }
    
        graph_chunks.clear();
    }

    void init_X_P(int layer_deep)
    {
        for (int i = 0; i < layer_deep; i++) {
            NtsVar d;
            X_P.push_back(d);
        }
    }
    
    int get_device_id(VertexId v_i) {
        for (int i = 0; i < device_num; i++) {
            if (v_i >= subdevice_offset[i] && v_i < subdevice_offset[i + 1]) {
                return i;
            }
        }
        printf("wrong vertex%d\n", v_i);
        assert(false);
    } 

    void GenerateAll(std::function<ValueType(VertexId, VertexId)> weight_compute,
                             VertexId *reorder_column_offset, VertexId *reorder_row_indices)
    {
        generatePartitionedSubgraph(reorder_column_offset, reorder_row_indices);
        std::cout << "finish generate partition subgraphs" << std::endl;
        PartitionToChunks(weight_compute);
        std::cout << "finish partition to chunks" << std::endl;
        generate_mirror_vertices(reorder_column_offset, reorder_row_indices);//for every chunk
        std::cout << "finish generate mirror vertices" << std::endl;
        generate_messige_index();
        std::cout << "finish generate message index" << std::endl;

        // generate_whole_graph_Topo();//for gat，
    }

    void generatePartitionedSubgraph(VertexId *reorder_column_offset, VertexId *reorder_row_indices)
    {
        printf("-------------------------------------device id : %d  \n\t\t  owned edges:%d, owned vertices:%d, global edges:%d, global vertices:%d, device_num:%d\nsubdevice_offset:", 
                global_device_id, owned_edges,owned_vertices,global_edges,global_vertices,device_num);
        for(int i = 0; i < device_num+1; i++)
        {
            std::cout << subdevice_offset[i] << " ";
        }
        std::cout << std::endl;

        this->dstList = new VertexId[owned_edges];
        this->srcList = new VertexId[owned_edges];
        int write_position=0; 
        for(VertexId local_id = 0; local_id < owned_vertices; local_id++)
        {
            VertexId dst = local_id + subdevice_offset[device_id];
            for(VertexId index = reorder_column_offset[dst]; index < reorder_column_offset[dst+1]; index++)
            {
                srcList[write_position] = reorder_row_indices[index];
                dstList[write_position++] = dst;
            }
        }

        // std::ofstream out("./log/cora_edgeList" + std::to_string(device_id) + ".txt", std::ios_base::out);
        // for(int i = 0; i < owned_edges; i++)
        // {
        //     out << dstList[i] << " " << srcList[i] << std::endl;
        // }

    }

    void PartitionToChunks(std::function<ValueType(VertexId, VertexId)> weight_compute)
    {
        graph_chunks.clear();
        std::vector<VertexId>edgecount;
        edgecount.resize(device_num,0);
        std::vector<VertexId>edgenumber;//本地边的src点在对应device的边数量
        edgenumber.resize(device_num,0);

        for(VertexId i=0;i<this->owned_edges;i++){
            VertexId src_partition=get_device_id(srcList[i]);
            edgenumber[src_partition]+=1;
        }

        for (VertexId i = 0; i < device_num; i++) {
            graph_chunks.push_back(new SubgraphSegment);
            graph_chunks[i]->init(subdevice_offset[i],
                                subdevice_offset[i + 1],
                                subdevice_offset[device_id],
                                subdevice_offset[device_id + 1],
                                edgenumber[i], is_CPU);
            graph_chunks[i]->allocVertexAssociateData();
            graph_chunks[i]->allocEdgeAssociateData();
        }
        std::vector<std::set<VertexId>> src_set;
        src_set.resize(device_num);
        for (VertexId i = 0; i < owned_edges; i++) {//设置graph chunk的src list和dst list
            int source = srcList[i];
            int destination = dstList[i];
            int src_partition = get_device_id(source);
            int offset = edgecount[src_partition]++;
            graph_chunks[src_partition]->source[offset] = source;
            graph_chunks[src_partition]->destination[offset] = destination;

            src_set[src_partition].insert(source);
        }
        //计算每一个chunk的source点，（去重）
        for(int i = 0; i < device_num; i++)
        {
            graph_chunks[i]->source_vertices.insert(graph_chunks[i]->source_vertices.end(), src_set[i].begin(), src_set[i].end());
        }
        VertexId *tmp_column_offset = new VertexId[global_vertices + 1];
        VertexId *tmp_row_offset = new VertexId[global_vertices + 1];
        for (VertexId i = 0; i < device_num; i++) {
            memset(tmp_column_offset, 0, sizeof(VertexId) * (global_vertices+ 1));
            memset(tmp_row_offset, 0, sizeof(VertexId) * (global_vertices + 1));
            for (VertexId j = 0; j < graph_chunks[i]->edge_size; j++) {
                //get offset（local id）
                VertexId v_src_m = graph_chunks[i]->source[j];
                VertexId v_dst_m = graph_chunks[i]->destination[j];
                VertexId v_dst = v_dst_m - graph_chunks[i]->dst_range[0];
                VertexId v_src = v_src_m - graph_chunks[i]->src_range[0];

                tmp_column_offset[v_dst + 1] += 1;
                
                tmp_row_offset[v_src + 1] += 1; 
            }
            //calc the partial sum
            graph_chunks[i]->column_offset[0] = 0;
            for (VertexId j = 0; j < graph_chunks[i]->batch_size_forward; j++) {
                tmp_column_offset[j + 1] += tmp_column_offset[j];
                graph_chunks[i]->column_offset[j + 1] = tmp_column_offset[j + 1];
            }
            graph_chunks[i]->row_offset[0]=0;
            for (VertexId j = 0; j < graph_chunks[i]->batch_size_backward; j++){
                tmp_row_offset[j + 1] += tmp_row_offset[j];
                graph_chunks[i]->row_offset[j + 1] = tmp_row_offset[j + 1];
            }
            //calc row indices
            for (VertexId j = 0; j < graph_chunks[i]->edge_size; j++) {
                // v_src is from partition i
                // v_dst is from local partition
                VertexId v_src_m = graph_chunks[i]->source[j];
                VertexId v_dst_m = graph_chunks[i]->destination[j];
                VertexId v_dst = v_dst_m - graph_chunks[i]->dst_range[0];
                VertexId v_src = v_src_m - graph_chunks[i]->src_range[0];
                graph_chunks[i]->row_indices[tmp_column_offset[v_dst]] = v_src_m;
                graph_chunks[i]->edge_weight_forward[tmp_column_offset[v_dst]++] =
                    weight_compute(v_src_m, v_dst_m);
                graph_chunks[i]->column_indices[tmp_row_offset[v_src]] = v_dst_m;
                graph_chunks[i]->edge_weight_backward[tmp_row_offset[v_src]++] =
                    weight_compute(v_src_m, v_dst_m);
                
            }
            for (VertexId j = 0; j < graph_chunks[i]->batch_size_forward; j++) {        
                // save the src and dst in the column format
                VertexId v_dst_m = j+ graph_chunks[i]->dst_range[0];
                for(VertexId e_idx=graph_chunks[i]->column_offset[j];e_idx<graph_chunks[i]->column_offset[j+1];e_idx++){
                    VertexId v_src_m = graph_chunks[i]->row_indices[e_idx];
                    graph_chunks[i]->source[e_idx] = (long)(v_src_m);
                    graph_chunks[i]->destination[e_idx]=(long)(v_dst_m);
                }
            }
        }

        delete[] tmp_column_offset;
        delete[] tmp_row_offset;
    }
    
    void generate_mirror_vertices(VertexId *reorder_column_offset, VertexId *reorder_row_indices)
    {
        // #pragma omp parallel for
        for(int i = 0; i < graph_chunks.size(); i++)
        {
            std::set<VertexId> mirror;
            VertexId v_start = graph_chunks[i]->src_range[0];
            VertexId v_end = graph_chunks[i]->src_range[1];
            for(VertexId v_id = v_start; v_id < v_end; v_id++)
            {
                for(int j = reorder_column_offset[v_id]; j < reorder_column_offset[v_id+1]; j++)
                {
                    VertexId v_neighber = reorder_row_indices[j];
                    if(device_id == get_device_id(v_neighber))
                    {
                        mirror.insert(v_neighber);
                    }
                }
            }
            graph_chunks[i]->mirror_vertices.insert(graph_chunks[i]->mirror_vertices.end(), mirror.begin(), mirror.end());
        }

        //print mirror
        // std::ofstream out("./log/cora_mirror"+std::to_string(global_device_id)+".txt", std::ios_base::out);
        // for(int i = 0; i < device_num; i++)
        // {
        //     out << "---------------------------------chunks " << i << std::endl;
        //     for(VertexId vtx = 0; vtx < graph_chunks[i]->mirror_vertices.size(); vtx++)
        //     {
        //         out << graph_chunks[i]->mirror_vertices[vtx] << " ";
        //     }
        //     out << std::endl;
        // }
    
        // print source
        // std::ofstream out("./log/cora_source"+std::to_string(global_device_id)+".txt", std::ios_base::out);
        // for(int i = 0; i < device_num; i++)
        // {
        //     out << "---------------------------------chunks " << i << std::endl;
        //     for(VertexId vtx = 0; vtx < graph_chunks[i]->source_vertices.size(); vtx++)
        //     {
        //         out << graph_chunks[i]->source_vertices[vtx] << " ";
        //     }
        //     out << std::endl;
        // }

    }

    void generate_messige_index()
    {
        for(int i = 0; i < device_num; i++)
        {
            graph_chunks[i]->generate_message_index();
        }

        // debug
        // std::ofstream out("./log/cora_message_index"+std::to_string(global_device_id)+".txt", std::ios_base::out);
        // for(int i = 0; i < device_num; i++)
        // {
        //     out << "-----------------device source: " << i <<" " << graph_chunks[i]->source_vertices.size() << std::endl;
        //     for(VertexId j = 0; j < graph_chunks[i]->source_vertices.size(); j++)
        //     {
        //         out << graph_chunks[i]->source_vertices[j] << "=>" << 
        //                     graph_chunks[i]->forward_message_index[graph_chunks[i]->source_vertices[j] - subdevice_offset[i]] << std::endl;;
        //     }
        //     // out << "-----------------device mirror: " << i  <<" " << graph_chunks[i]->mirror_vertices.size() << std::endl;
        //     // for(int j = 0; j < graph_chunks[i]->mirror_vertices.size(); j++)
        //     // {
        //     //     out << j << "=>" << graph_chunks[i]->backward_message_index[j] + subdevice_offset[device_id] << std::endl;;
        //     // }
        //     // out << std::endl;
        // }
        // out.close();

    }
    
    void generate_source_index(){
      source_vtx=new VertexId[global_vertices+1];
      memset(source_vtx,0,sizeof(VertexId)*(global_vertices+1));
      for(VertexId i=0;i<this->owned_edges;i++){
          source_vtx[srcList[i]+1]=1;
      }
      for(VertexId i=0;i<global_vertices;i++){
          source_vtx[i+1]+=source_vtx[i];
      }
      owned_sources=source_vtx[global_vertices];
  }
    
    void generate_whole_graph_Topo()
    {
        generate_source_index();
        column_offset=new VertexId[owned_vertices+1];
        row_offset=new VertexId[owned_sources+1];
        column_indices=new VertexId[owned_edges];
        row_indices=new VertexId[owned_edges];
        memset(column_offset,0,(owned_vertices+1)*sizeof(VertexId));
        memset(row_offset,0,(owned_sources+1)*sizeof(VertexId));
        memset(column_indices,0,owned_edges*sizeof(VertexId));
        memset(row_indices,0,owned_edges*sizeof(VertexId));
        VertexId* tmp_column_offset=new VertexId[owned_vertices+1]; 
        VertexId* tmp_row_offset=new VertexId[owned_sources+1];
        memset(tmp_column_offset,0,(owned_vertices+1)*sizeof(VertexId));
        memset(tmp_row_offset,0,(owned_sources+1)*sizeof(VertexId));
        for(VertexId i=0;i<this->owned_edges;i++){
            VertexId src_trans=this->source_vtx[this->srcList[i]];//第几个source点
            VertexId dst_trans=this->dstList[i]-subdevice_offset[device_id];
            tmp_column_offset[dst_trans+1]++;
            tmp_row_offset[src_trans+1]++;
        }
        for(int i=0;i<owned_vertices;i++){
            tmp_column_offset[i+1]+=tmp_column_offset[i];
        }
        for(int i=0;i<owned_sources;i++){
            tmp_row_offset[i+1]+=tmp_row_offset[i];
        }
        memcpy(column_offset,tmp_column_offset,sizeof(VertexId)*(owned_vertices+1));
        memcpy(row_offset,tmp_row_offset,sizeof(VertexId)*(owned_sources+1));
        for(VertexId i=0;i<this->owned_edges;i++){
            VertexId src=this->srcList[i];
            VertexId dst=this->dstList[i];
            VertexId src_trans=this->source_vtx[src];
            VertexId dst_trans=dst-subdevice_offset[device_id];
            row_indices[tmp_column_offset[dst_trans]++]=src;
            column_indices[tmp_row_offset[src_trans]++]=dst;
        }
        delete []tmp_column_offset;
        delete []tmp_row_offset;
    }
    
    void load_feature(ValueType *reorder_feat)
    {
        // std::cout << "start load" << std::endl;
        feature = graph->Nts->NewLeafTensor(reorder_feat + subdevice_offset[device_id] * feature_size, {owned_vertices, feature_size}, 
                torch::DeviceType::CPU);
        // std::cout << "end load feature" << std::endl;
    }
    void load_label_mask(long *reorder_label, int *reorder_mask)
    {
        mask = graph->Nts->NewLeafKIntTensor(reorder_mask + subdevice_offset[device_id], {owned_vertices, 1});
        // std::cout << "end load mask" << std::endl;
        label = graph->Nts->NewLeafKLongTensor(reorder_label + subdevice_offset[device_id], {owned_vertices});
        // std::cout << "end load label" << std::endl;

        //debug
        // std::ofstream out("./log/cora_flm_" + std::to_string(global_device_id) + ".txt", std::ios_base::out);
        // for(VertexId id = 0; id < owned_vertices; id++){
        //     out <<"reorder dst id: " << id + subdevice_offset[global_device_id];
        //     out << " label: " << label.data<long>()[id];
        //     out << " mask: " << mask.data<int>()[id];
        //     out << " feature: " ;
        //     for(int i = 0; i < feature_size; i++)
        //     {
        //         out << feature[id].data<float>()[i] << " ";
        //     } 
        //     out << std::endl;
        // }
        // out.close();

    }

    void init_CudaStream()
    {
        cudaSetUsingDevice(global_device_id);
        cudaStream_t cuda_stream;
        cudaStreamCreateWithFlags(&(cuda_stream), cudaStreamNonBlocking);

        cs = new Cuda_Stream();
        cs->setNewStream(cuda_stream);
        // ts = at::cuda::CUDAStream(at::cuda::CUDAStream::UNCHECKED, 
        //                         at::Stream(at::Stream::UNSAFE, 
        //                         c10::Device(at::DeviceType::CUDA, global_device_id),
        //                         reinterpret_cast<int64_t>(cuda_stream)));
        
        ts = new at::cuda::CUDAStream(at::cuda::CUDAStream::UNCHECKED, 
                                at::Stream(at::Stream::UNSAFE, 
                                c10::Device(at::DeviceType::CUDA, global_device_id),
                                reinterpret_cast<int64_t>(cuda_stream)));
    }

    void load_data_to_GPUs()
    {
        cudaSetUsingDevice(global_device_id);

        // std::cout << "device id[" << global_device_id << "] label size: " << label.sizes() << std::endl;
        // std::cout << "device id[" << global_device_id << "] feature size: " << feature.sizes() << std::endl;

        X[0] = feature.cuda().set_requires_grad(true);
        label_gpu = label.cuda();
        mask_gpu = mask.cuda();
        
        //parameter
        torch::Device GPU(torch::kCUDA, global_device_id);
        for(int i = 0; i < P.size(); i++)
        {
            P[i]->to(GPU);
            P[i]->Adam_to_GPU(global_device_id);

            //现在的写法是每个GPU都做P和T，所以每个GPU都有P、feature和图拓扑
        }

        //图结构
        for(int i = 0; i < graph_chunks.size(); i++)
        {
            graph_chunks[i]->load_graph_to_GPUs(cs->stream);
        }
        
        
        LOG_INFO("finish load data to multi GPU");
    }

    void load_graph_data_T_before_P_to_GPU()
    {
        cudaSetUsingDevice(global_device_id);

        label_gpu = label.cuda();
        mask_gpu = mask.cuda();
        
        //图结构
        for(int i = 0; i < graph_chunks.size(); i++)
        {
            graph_chunks[i]->load_graph_to_GPUs(cs->stream);
        }
        
        printf("decouple(T before P) device [%d] finish load graph data to multi GPU\n", global_device_id);
    }

    void load_NN_data_T_before_P_to_GPU()
    {
        cudaSetUsingDevice(global_device_id);

        X[0] = feature.cuda().set_requires_grad(true);
        
        //parameter
        torch::Device GPU(torch::kCUDA, global_device_id);
        for(int i = 0; i < P.size(); i++)
        {
            P[i]->to(GPU);
            P[i]->Adam_to_GPU(global_device_id);
        }
        
        printf("decouple(T before P) device [%d] finish load NN data to multi GPU\n", global_device_id);
    }

    void load_NN_data_T_before_P_to_GPU_without_Para()
    {
        cudaSetUsingDevice(global_device_id);

        X[0] = feature.cuda().set_requires_grad(true);
        
        printf("decouple(T before P) device [%d] finish load NN data to multi GPU\n", global_device_id);
    }
    
    void init_graph_communicator(NCCL_Communicator *nccl_comm_)
    {
        VertexId *source_ = new VertexId[device_num];
        VertexId *mirror_ = new VertexId[device_num];
        for(int i = 0; i < device_num; i++)
        {
            source_[i] = graph_chunks[i]->source_vertices.size();
            mirror_[i] = graph_chunks[i]->mirror_vertices.size();
            // printf("device[%d]------> source[%d]:%d\tmirror[%d]:%d\n", global_device_id, i, source_[i], i, mirror_[i]);
        }
        nccl_graph_comm = new NCCL_graph_Communicator(global_device_id, subdevice_offset, device_num, source_, mirror_, nccl_comm_, cs);
    }

    void set_PT_Comm_message(int all_device_num_, int P_num_)
    {
        all_device_num = all_device_num_;
        P_num = P_num_;
    }
    
    void init_PT_communicator(NCCL_Communicator *nccl_comm_)//P与T之间传递embedding用的
    {
        VertexId * msg_offset;
        if(root == 0)
        {
            msg_offset = P2T_offset;
        }
        else
        {
            msg_offset = T2P_offset;
        }
        nccl_pt_comm = new NCCL_PT_Communicator(global_device_id, all_device_num, P_num, nccl_comm_, cs, msg_offset);
    }
    
    
    
    //图传播部分直接使用device id就行，就不改了，因为默认图操作的root = 0
    /*
    先发送消息（多GPU通信）获取embedding，再进行计算
    */
    void sync_and_compute(ValueType *f_input_gpu, int embedding_size,  ValueType *f_output_gpu) 
    {
        //没有测试，应该输出一下f_input（在GPU）的内容，和发送接收的信息

        nccl_graph_comm->init_nccl_layer_all_full_graph(embedding_size, sendmirror);

        {
            //1-stage
            //put data to send buffer
            //第一点，这里后期需要优化，写一个cuda核函数将数据放入send buffer，速度肯定比这个快
            //第二点，这里是full graph训练，给本地发送的时候不需要转到本地的buffer，直接帮忙存一下指针就行
            // double emit_buffer_time = 0.0;
            // emit_buffer_time -= get_time();
            for(int step = 0; step < device_num; step++)
            {
                int trigger_device = step;
                nccl_graph_comm->set_current_send_device(trigger_device);
                // LOG_INFO("1");
                if(trigger_device != device_id) // 本地？
                {//full graph训练， 不需要转到本地的buffer，这是因为graph_chunks[device_id]->send_vtx.size() == owned_vertices
                    // std::cout<<"device_id trigger_id size"<<device_id<<' '<<trigger_device<<' '<<graph_chunks[trigger_device]->mirror_vertices.size()<<std::endl;
                    const auto &send_vtx_cpu = graph_chunks[trigger_device]->mirror_vertices;//mirror_vertices的size,

                    // const auto &send_vtx = graph_chunks[trigger_device]->mirror_vertices_gpu; //不传vtx id的话就用不到这个
        
                    VertexId dst_start = graph_chunks[trigger_device]->dst_range[0];
                    VertexId dst_end = graph_chunks[trigger_device]->dst_range[1];

                    // int num_t = int((max_threads-device_num)/device_num);
                    // printf("max_threads[%d]  num_thread:    %d\n",max_threads, num_t);
                    // printf("send_vtx_cpu[%d->%d]  num vtx:    %d\n",device_id, trigger_device, send_vtx_cpu.size());
                    // #pragma omp parallel num_threads(num_t)
                    for(int i = 0; i < send_vtx_cpu.size(); i++)
                    {
                        VertexId local_mirror = send_vtx_cpu[i] - dst_start;
                        // nccl_graph_comm->emit_buffer(f_input_gpu+local_mirror*embedding_size, send_vtx+i, i);
                        nccl_graph_comm->emit_buffer_only_embedding(f_input_gpu+local_mirror*embedding_size, i);
                    }
                }
                else
                {
                    nccl_graph_comm->store_input_full_graph(f_input_gpu);
                }
                nccl_graph_comm->trigger_one_device(trigger_device);

            }
            // nccl_graph_comm->debug();
            
            cs->CUDA_DEVICE_SYNCHRONIZE();
            // emit_buffer_time += get_time();
            // std::cout << "emit time from g comp: " << emit_buffer_time << std::endl;

            //2-stage send and receive
            //后期需要优化，在send buffer之前开启线程追踪send、recv queue的情况，一有东西直接发送
            //这样可以将2-stage去掉
            // double p2p_time = 0.0;
            // p2p_time -= get_time();

            nccl_graph_comm->point_to_point();
            cs->CUDA_DEVICE_SYNCHRONIZE();

            double gpus_aggregate_time = 0.0;
            gpus_aggregate_time-= get_time();
            for(int step = 0; step < device_num; step++)
            {
                VertexId trigger_device = -1;
                ValueType *source_info_gpu;
                source_info_gpu = nccl_graph_comm->receive_one_device_full_graph(trigger_device, step);
                
                VertexId *col_offset_gpu = graph_chunks[trigger_device]->column_offset_gpu;
                VertexId *row_indi_gpu = graph_chunks[trigger_device]->row_indices_gpu;
                ValueType *weight_gpu = graph_chunks[trigger_device]->edge_weight_forward_gpu;
                VertexId *index_gpu = graph_chunks[trigger_device]->forward_message_index_gpu;

                VertexId src_start = graph_chunks[trigger_device]->src_range[0];
                VertexId src_end = graph_chunks[trigger_device]->src_range[1];
                VertexId dst_start = graph_chunks[trigger_device]->dst_range[0];
                VertexId dst_end = graph_chunks[trigger_device]->dst_range[1];

                VertexId forword_batch_size = graph_chunks[trigger_device]->batch_size_forward;

                cs->Gather_By_Dst_From_Src_with_index(
                    source_info_gpu, f_output_gpu, weight_gpu,
                    row_indi_gpu, col_offset_gpu, 
                    src_start, src_end, dst_start, dst_end,
                    graph_chunks[trigger_device]->source_vertices.size(), graph_chunks[trigger_device]->edge_size,//for debug
                    forword_batch_size, embedding_size, index_gpu, graph->rtminfo->with_weight
                );

            }
            CHECK_CUDA_RESULT(cudaDeviceSynchronize());
            cs->CUDA_DEVICE_SYNCHRONIZE();
            gpus_aggregate_time+= get_time();
            std::cout << "gpus_aggregate_time: " << gpus_aggregate_time << std::endl;
            // agg_time += get_time();
            // std::cout << "agg time from g comp: " << agg_time << std::endl;
        }
        nccl_graph_comm->release_nccl_graph_comm();


    }


    void compute_and_sync_backward(ValueType *f_input_gpu, int grad_size,  ValueType *f_output_gpu)
    {
        nccl_graph_comm->init_nccl_layer_all_full_graph(grad_size, sendsource);

        {

            for(int step = 0; step < device_num; step++)
            {
                int trigger_device = (step + 1 + device_id) % device_num;

                ValueType *source_embedding = (ValueType*)cudaMallocGPU(
                                    sizeof(ValueType) * grad_size * graph_chunks[trigger_device]->source_vertices.size(),
                                    cs->stream);
                VertexId *row_offset_gpu = graph_chunks[trigger_device]->row_offset_gpu;
                VertexId *col_indices_gpu = graph_chunks[trigger_device]->column_indices_gpu;
                VertexId *source_idx_gpu = graph_chunks[trigger_device]->forward_message_index_gpu;
                ValueType *weight_gpu = graph_chunks[trigger_device]->edge_weight_backward_gpu;

                VertexId src_start = graph_chunks[trigger_device]->src_range[0];
                VertexId src_end = graph_chunks[trigger_device]->src_range[1];
                VertexId dst_start = graph_chunks[trigger_device]->dst_range[0];
                VertexId dst_end = graph_chunks[trigger_device]->dst_range[1];

                VertexId backward_batch_size = graph_chunks[trigger_device]->batch_size_backward;


                cs->Gather_By_Src_From_Dst_with_index(
                    f_input_gpu, source_embedding, weight_gpu, 
                    col_indices_gpu, row_offset_gpu,
                    src_start, src_end, dst_start, dst_end,
                    graph_chunks[trigger_device]->source_vertices.size(), graph_chunks[trigger_device]->edge_size,//for debug
                    backward_batch_size, grad_size, source_idx_gpu, graph->rtminfo->with_weight
                );

                cs->CUDA_DEVICE_SYNCHRONIZE();
                
                   

                nccl_graph_comm->set_current_send_device(trigger_device);

                if(trigger_device != device_id)
                {
                    const auto &send_vtx_cpu = graph_chunks[trigger_device]->source_vertices;

                    for(int i = 0; i < send_vtx_cpu.size(); i++)
                    {
                        VertexId local_source = send_vtx_cpu[i] - src_start;
                        assert(i == graph_chunks[trigger_device]->forward_message_index[local_source]);

                        nccl_graph_comm->emit_buffer_only_embedding(source_embedding + i * grad_size, i);
                    }
                } else {
                    nccl_graph_comm->store_input_full_graph(f_input_gpu);
                }
                nccl_graph_comm->trigger_one_device(trigger_device);
                cudaFreeGPU(source_embedding, cs->stream);
            }


            nccl_graph_comm->point_to_point();


            cudaDeviceSynchronize();

            //3-stage merge receive grad
            for(int step = 0; step < device_num; step++)
            {
                VertexId trigger_device = -1;
                ValueType *source_info_gpu;
                source_info_gpu = nccl_graph_comm->receive_one_device_full_graph(trigger_device, step);

                VertexId *index = graph_chunks[trigger_device]->backward_message_index_gpu;

                cs->merge_data_grad_with_index(
                    source_info_gpu, f_output_gpu, graph_chunks[trigger_device]->mirror_vertices.size(),
                    grad_size, index
                );
                    //debug
                    // ValueType *source_embedding_cpu = (ValueType*)malloc(
                    //                 sizeof(ValueType) * grad_size * graph_chunks[trigger_device]->mirror_vertices.size());
                    // CHECK_CUDA_RESULT(cudaMemcpy(source_embedding_cpu, source_info_gpu, 
                    //                         size_of_msg(grad_size) * graph_chunks[trigger_device]->mirror_vertices.size(),
                    //                         cudaMemcpyDeviceToHost));
                    // out << "----------------trigger device : " << trigger_device << std::endl;
                    // for(int i = 0; i < graph_chunks[trigger_device]->mirror_vertices.size(); i++)
                    // {
                    //     out << "dst id[" << graph_chunks[trigger_device]->mirror_vertices[i] << "] grad : ";
                    //     for(int j = 0; j < grad_size; j++)
                    //     {
                    //         out << j << ":" << source_embedding_cpu[i*grad_size + j] << " ";
                    //     }
                    //     out << std::endl;
                    // }
                    // ValueType *f_output_cpu = (ValueType*)malloc(
                    // sizeof(ValueType) * grad_size * owned_vertices);
                    // CHECK_CUDA_RESULT(cudaMemcpy(f_output_cpu, f_output_gpu, 
                    //                         size_of_msg(grad_size) *owned_vertices,
                    //                         cudaMemcpyDeviceToHost));
                    // out << "----------------step : " << step << std::endl;
                    // for(int i = 0; i < owned_vertices; i++)
                    // {
                    //     out << "merge dst id[" << i + subdevice_offset[device_id] << "] grad : ";
                    //     for(int j = 0; j < grad_size; j++)
                    //     {
                    //         out << j << ":" << f_output_cpu[i*grad_size + j] << " ";
                    //     }
                    //     out << std::endl;
                    // }
            }
        }
        
        cs->CUDA_DEVICE_SYNCHRONIZE();
        nccl_graph_comm->release_nccl_graph_comm();


    }

    void sync_and_compute_SpMM(ValueType *f_input_gpu, int embedding_size,  ValueType *f_output_gpu)
    {
        nccl_graph_comm->init_nccl_layer_all_full_graph(embedding_size, sendmirror);

        {

            #pragma omp parallel for
            for(int step = 0; step < device_num; step++)
            {
                int trigger_device = step;
                nccl_graph_comm->set_current_send_device(trigger_device);

                if(trigger_device != device_id)
                {

                    std::cout<<"device_id trigger_divice size"<<device_id<<' '<<trigger_device<<' '<<graph_chunks[trigger_device]->mirror_vertices.size()<<std::endl;
                    nccl_graph_comm->emit_buffer_only_embedding(f_input_gpu, 0, graph_chunks[trigger_device]->mirror_vertices.size());
                }
                else
                {
                    nccl_graph_comm->store_input_full_graph(f_input_gpu);
                }
                nccl_graph_comm->trigger_one_device(trigger_device);

            }

            cs->CUDA_DEVICE_SYNCHRONIZE();



            nccl_graph_comm->point_to_point();
            cs->CUDA_DEVICE_SYNCHRONIZE();

            for(int step = 0; step < device_num; step++)
            {
                VertexId trigger_device = -1;
                ValueType *source_info_gpu;
                source_info_gpu = nccl_graph_comm->receive_one_device_full_graph(trigger_device, step);
                
                VertexId *col_offset_gpu = graph_chunks[trigger_device]->column_offset_gpu;
                VertexId *row_indi_gpu = graph_chunks[trigger_device]->row_indices_gpu;
                ValueType *weight_gpu = graph_chunks[trigger_device]->edge_weight_forward_gpu;
                VertexId *index_gpu = graph_chunks[trigger_device]->forward_message_index_gpu;

                VertexId src_start = graph_chunks[trigger_device]->src_range[0];
                VertexId src_end = graph_chunks[trigger_device]->src_range[1];
                VertexId dst_start = graph_chunks[trigger_device]->dst_range[0];
                VertexId dst_end = graph_chunks[trigger_device]->dst_range[1];

                VertexId forword_batch_size = graph_chunks[trigger_device]->batch_size_forward;

                cs->Gather_By_Dst_From_Src_with_index_spmm(
                    source_info_gpu, f_output_gpu, weight_gpu,
                    row_indi_gpu, col_offset_gpu, src_end - src_start,
                    src_start, src_end, dst_start, dst_end,
                    graph_chunks[trigger_device]->source_vertices.size(), graph_chunks[trigger_device]->edge_size,//for debug
                    forword_batch_size, embedding_size, index_gpu, graph->rtminfo->with_weight
                );
            }

            cs->CUDA_DEVICE_SYNCHRONIZE();
        }
        nccl_graph_comm->release_nccl_graph_comm();


    }


    void compute_and_sync_backward_SpMM(ValueType *f_input_gpu, int grad_size,  ValueType *f_output_gpu)
    {
        nccl_graph_comm->init_nccl_layer_all_full_graph(grad_size, sendsource);

            for(int step = 0; step < device_num; step++)
            {
                int trigger_device = (step + 1 + device_id) % device_num;

                ValueType *source_embedding = (ValueType*)cudaMallocGPU(
                                    sizeof(ValueType) * grad_size * graph_chunks[trigger_device]->source_vertices.size(),
                                    cs->stream);
                VertexId *row_offset_gpu = graph_chunks[trigger_device]->row_offset_gpu;
                VertexId *col_indices_gpu = graph_chunks[trigger_device]->column_indices_gpu;
                VertexId *source_idx_gpu = graph_chunks[trigger_device]->forward_message_index_gpu;
                ValueType *weight_gpu = graph_chunks[trigger_device]->edge_weight_backward_gpu;

                VertexId src_start = graph_chunks[trigger_device]->src_range[0];
                VertexId src_end = graph_chunks[trigger_device]->src_range[1];
                VertexId dst_start = graph_chunks[trigger_device]->dst_range[0];
                VertexId dst_end = graph_chunks[trigger_device]->dst_range[1];

                VertexId backward_batch_size = graph_chunks[trigger_device]->batch_size_backward;



                // cs->Gather_By_Src_From_Dst_with_index(
                //     f_input_gpu, source_embedding, weight_gpu, 
                //     col_indices_gpu, row_offset_gpu,
                //     src_start, src_end, dst_start, dst_end,
                //     graph_chunks[trigger_device]->source_vertices.size(), graph_chunks[trigger_device]->edge_size,//for debug
                //     backward_batch_size, grad_size, source_idx_gpu, graph->rtminfo->with_weight
                // );

                cs->Gather_By_Src_From_Dst_with_index_spmm(
                    f_input_gpu, source_embedding, weight_gpu, 
                    col_indices_gpu, row_offset_gpu, dst_end - dst_start,
                    src_start, src_end, dst_start, dst_end,
                    graph_chunks[trigger_device]->source_vertices.size(), graph_chunks[trigger_device]->edge_size,//for debug
                    backward_batch_size, grad_size, source_idx_gpu, graph->rtminfo->with_weight
                );

                cs->CUDA_DEVICE_SYNCHRONIZE();
                


                nccl_graph_comm->set_current_send_device(trigger_device);

                if(trigger_device != device_id)
                {

                    nccl_graph_comm->emit_buffer_only_embedding(f_input_gpu, 0, graph_chunks[trigger_device]->source_vertices.size());
                } else {
                    nccl_graph_comm->store_input_full_graph(f_input_gpu);
                }
                nccl_graph_comm->trigger_one_device(trigger_device);
                cudaFreeGPU(source_embedding, cs->stream);
            }

            //2-stage send and receive grad
            nccl_graph_comm->point_to_point();
            // nccl_graph_comm->comm_only_embedding_debug(); // debug recv buffer's info
            // out << "finish test commm grad" << std::endl;


            //3-stage merge receive grad
            for(int step = 0; step < device_num; step++)
            {
                VertexId trigger_device = -1;
                ValueType *source_info_gpu;
                source_info_gpu = nccl_graph_comm->receive_one_device_full_graph(trigger_device, step);

                VertexId *index = graph_chunks[trigger_device]->backward_message_index_gpu;

                cs->merge_data_grad_with_index(
                    source_info_gpu, f_output_gpu, graph_chunks[trigger_device]->mirror_vertices.size(),
                    grad_size, index
                );
            }
        }
        
        cs->CUDA_DEVICE_SYNCHRONIZE();
        nccl_graph_comm->release_nccl_graph_comm();


    }

    void bcast_and_compute_SpMM(ValueType *f_input_gpu, int embedding_size,  ValueType *f_output_gpu)
    {
        //debug
        std::ofstream out("./log/debug_before_broadcast" + std::to_string(device_id) + "_f_input.txt", std::ios_base::out);
        ValueType *source_embedding_cpu = (ValueType*)malloc(
                        sizeof(ValueType) * embedding_size * owned_vertices);
        CHECK_CUDA_RESULT(cudaMemcpy(source_embedding_cpu, f_input_gpu, 
                                size_of_msg(embedding_size) * owned_vertices,
                                cudaMemcpyDeviceToHost));
        for(int i = 0; i < owned_vertices; i++)
        {
            out << "src id[" << i + subdevice_offset[device_id] << "] embedding : ";
            for(int j = 0; j < embedding_size; j++)
            {
                out << j << ":" << source_embedding_cpu[i*embedding_size + j] << " ";
            }
            out << std::endl;
        }
        out.close();

        nccl_graph_comm->init_nccl_layer_all_full_graph(embedding_size, bcast);
        {
            //1-stage
            //broadcast the input to all devices
            //2-stage compute with SpMM
            for(int step = 0; step < device_num; step++)
            {
                VertexId trigger_device = step;
                
                VertexId *col_offset_gpu = graph_chunks[trigger_device]->column_offset_gpu;
                VertexId *row_indi_gpu = graph_chunks[trigger_device]->row_indices_gpu;
                ValueType *weight_gpu = graph_chunks[trigger_device]->edge_weight_forward_gpu;
                VertexId *index_gpu = graph_chunks[trigger_device]->forward_message_index_gpu;

                VertexId src_start = graph_chunks[trigger_device]->src_range[0];
                VertexId src_end = graph_chunks[trigger_device]->src_range[1];
                VertexId dst_start = graph_chunks[trigger_device]->dst_range[0];
                VertexId dst_end = graph_chunks[trigger_device]->dst_range[1];

                VertexId dst_num = graph_chunks[trigger_device]->batch_size_forward;
                VertexId src_num = graph_chunks[trigger_device]->batch_size_backward;

                ValueType *source_info_gpu = (ValueType*) cudaMallocGPU(
                                src_num * size_of_msg(feature_size), cs->stream);
                nccl_graph_comm->broadcast(f_input_gpu, source_info_gpu, embedding_size, owned_vertices, step);
                nccl_graph_comm->debug_broadcast(f_input_gpu, src_num, step);

            }

            // sleep(5);
            assert(false);

            cs->CUDA_DEVICE_SYNCHRONIZE();
        }
        nccl_graph_comm->release_nccl_graph_comm();


    }

    //反向的SpMM可以使用CSR直接干就完了。// 累加结果！！！
    void bcast_and_compute_backward_SpMM(ValueType *f_input_gpu, int grad_size,  ValueType *f_output_gpu)
    {
        nccl_graph_comm->init_nccl_layer_all_full_graph(grad_size, sendsource);
        // std::ofstream out("./log/cora_receive_info_" + std::to_string(device_id) + ".txt", std::ios_base::out);//for debug

        // out << "-----------------start graph backward : " << std::endl;

        {
            //1-stage compute grad to source and put source to buffer
            for(int step = 0; step < device_num; step++)
            {
                int trigger_device = (step + 1 + device_id) % device_num;

                ValueType *source_embedding = (ValueType*)cudaMallocGPU(
                                    sizeof(ValueType) * grad_size * graph_chunks[trigger_device]->source_vertices.size(),
                                    cs->stream);
                VertexId *row_offset_gpu = graph_chunks[trigger_device]->row_offset_gpu;
                VertexId *col_indices_gpu = graph_chunks[trigger_device]->column_indices_gpu;
                VertexId *source_idx_gpu = graph_chunks[trigger_device]->forward_message_index_gpu;
                ValueType *weight_gpu = graph_chunks[trigger_device]->edge_weight_backward_gpu;

                VertexId src_start = graph_chunks[trigger_device]->src_range[0];
                VertexId src_end = graph_chunks[trigger_device]->src_range[1];
                VertexId dst_start = graph_chunks[trigger_device]->dst_range[0];
                VertexId dst_end = graph_chunks[trigger_device]->dst_range[1];

                VertexId backward_batch_size = graph_chunks[trigger_device]->batch_size_backward;

                // out << "start gather by src from dst [trigger_device]: " << trigger_device << std::endl;

                cs->Gather_By_Src_From_Dst_with_index(
                    f_input_gpu, source_embedding, weight_gpu, 
                    col_indices_gpu, row_offset_gpu,
                    src_start, src_end, dst_start, dst_end,
                    graph_chunks[trigger_device]->source_vertices.size(), graph_chunks[trigger_device]->edge_size,//for debug
                    backward_batch_size, grad_size, source_idx_gpu, graph->rtminfo->with_weight
                );

                cs->CUDA_DEVICE_SYNCHRONIZE();
                
                    //debug
                    // ValueType *source_embedding_cpu = (ValueType*)malloc(
                    //                 sizeof(ValueType) * grad_size * graph_chunks[trigger_device]->source_vertices.size());
                    // CHECK_CUDA_RESULT(cudaMemcpy(source_embedding_cpu, source_embedding, 
                    //                         size_of_msg(grad_size) * graph_chunks[trigger_device]->source_vertices.size(),
                    //                         cudaMemcpyDeviceToHost));
                    // for(int i = 0; i < graph_chunks[trigger_device]->source_vertices.size(); i++)
                    // {
                    //     out << "src id[" << graph_chunks[trigger_device]->source_vertices[i] << "] grad : ";
                    //     for(int j = 0; j < grad_size; j++)
                    //     {
                    //         out << j << ":" << source_embedding_cpu[i*grad_size + j] << " ";
                    //     }
                    //     out << std::endl;
                    // }

                nccl_graph_comm->set_current_send_device(trigger_device);

                if(trigger_device != device_id)
                {
                    const auto &send_vtx_cpu = graph_chunks[trigger_device]->source_vertices;

                    for(int i = 0; i < send_vtx_cpu.size(); i++)
                    {
                        VertexId local_source = send_vtx_cpu[i] - src_start;
                        assert(i == graph_chunks[trigger_device]->forward_message_index[local_source]);

                        nccl_graph_comm->emit_buffer_only_embedding(source_embedding + i * grad_size, i);
                    }
                } else {
                    nccl_graph_comm->store_input_full_graph(f_input_gpu);
                }
                nccl_graph_comm->trigger_one_device(trigger_device);
                cudaFreeGPU(source_embedding, cs->stream);
            }

            //2-stage send and receive grad
            nccl_graph_comm->point_to_point();
            // nccl_graph_comm->comm_only_embedding_debug(); // debug recv buffer's info
            // out << "finish test commm grad" << std::endl;


            //3-stage merge receive grad
            for(int step = 0; step < device_num; step++)
            {
                VertexId trigger_device = -1;
                ValueType *source_info_gpu;
                source_info_gpu = nccl_graph_comm->receive_one_device_full_graph(trigger_device, step);

                VertexId *index = graph_chunks[trigger_device]->backward_message_index_gpu;

                cs->merge_data_grad_with_index(
                    source_info_gpu, f_output_gpu, graph_chunks[trigger_device]->mirror_vertices.size(),
                    grad_size, index
                );
                    //debug
                    // ValueType *source_embedding_cpu = (ValueType*)malloc(
                    //                 sizeof(ValueType) * grad_size * graph_chunks[trigger_device]->mirror_vertices.size());
                    // CHECK_CUDA_RESULT(cudaMemcpy(source_embedding_cpu, source_info_gpu, 
                    //                         size_of_msg(grad_size) * graph_chunks[trigger_device]->mirror_vertices.size(),
                    //                         cudaMemcpyDeviceToHost));
                    // out << "----------------trigger device : " << trigger_device << std::endl;
                    // for(int i = 0; i < graph_chunks[trigger_device]->mirror_vertices.size(); i++)
                    // {
                    //     out << "dst id[" << graph_chunks[trigger_device]->mirror_vertices[i] << "] grad : ";
                    //     for(int j = 0; j < grad_size; j++)
                    //     {
                    //         out << j << ":" << source_embedding_cpu[i*grad_size + j] << " ";
                    //     }
                    //     out << std::endl;
                    // }
                    // ValueType *f_output_cpu = (ValueType*)malloc(
                    // sizeof(ValueType) * grad_size * owned_vertices);
                    // CHECK_CUDA_RESULT(cudaMemcpy(f_output_cpu, f_output_gpu, 
                    //                         size_of_msg(grad_size) *owned_vertices,
                    //                         cudaMemcpyDeviceToHost));
                    // out << "----------------step : " << step << std::endl;
                    // for(int i = 0; i < owned_vertices; i++)
                    // {
                    //     out << "merge dst id[" << i + subdevice_offset[device_id] << "] grad : ";
                    //     for(int j = 0; j < grad_size; j++)
                    //     {
                    //         out << j << ":" << f_output_cpu[i*grad_size + j] << " ";
                    //     }
                    //     out << std::endl;
                    // }
            }
        }
        
        cs->CUDA_DEVICE_SYNCHRONIZE();
        nccl_graph_comm->release_nccl_graph_comm();


    }

    //decouple version
    void send_emb_from_T_to_P_device()
    {
        cudaSetUsingDevice(global_device_id);
        // LOG_INFO("global device id [%d]", global_device_id);
        int embedding_size = X[X.size() - 1].size(1);

        //print T2P_offset
        // std::cout << "send offset[" << global_device_id << "]->T2P_offset : ";
        // for(int j = 0; j < P_num + 1; j++)
        // {
        //     std::cout << T2P_offset[j] << " ";
        // }
        // std::cout << std::endl;
        
        // nccl_pt_comm->init_nccl_send_to_P(embedding_size);

        ValueType *f_input_buffer =
                graph->Nts->getWritableBuffer(X[X.size() - 1], torch::DeviceType::CUDA);

        nccl_pt_comm->T2P_send_from_X_buffer(embedding_size, f_input_buffer);

    }

    void recv_emb_from_T_to_P_device(int embedding_size)
    {
        cudaSetUsingDevice(global_device_id);
        // LOG_INFO("global device id [%d]", global_device_id);

        NtsVar embedding = graph->Nts->NewKeyTensor({owned_vertices, embedding_size}, 
                torch::DeviceType::CUDA, global_device_id);
        X[0] = embedding;

        // printf("global device id[%d] : X[0] device[%d]、 X[0] size{%d, %d}\n", global_device_id, X[0].device(), X[0].size(0), X[0].size(1));
        // std::cout << X[0].device() << std::endl;

        ValueType *f_output_buffer=
                graph->Nts->getWritableBuffer(X[0], torch::DeviceType::CUDA);

        nccl_pt_comm->T2P_recv_write_to_X_buffer(embedding_size, f_output_buffer);


    }

    void send_grad_from_P_to_T_device()
    {
        cudaSetUsingDevice(global_device_id);

        int embedding_size = decoupled_mid_grad.size(1);

        ValueType *f_input_buffer =
                graph->Nts->getWritableBuffer(decoupled_mid_grad, torch::DeviceType::CUDA);

        nccl_pt_comm->P2T_send_from_X_buffer(embedding_size, f_input_buffer);

            // std::ofstream out("./log/cora_send_grad_"+ std::to_string(global_device_id) + ".txt", std::ios_base::out);//for debug
            // out << "(out data)GPU ID: " << decoupled_mid_grad.device() << std::endl;
            // out << "(out data)size: " << decoupled_mid_grad.sizes() << std::endl;
            // NtsVar X_layer = decoupled_mid_grad.to(torch::DeviceType::CPU);
            // for(int i = 0; i < owned_vertices; i++)
            // {
            //     out << "dst id[" << i + subdevice_offset[device_id] << "] embedding: ";
            //     for(int j = 0; j < X_layer.size(1); j++)
            //     {
            //         out << X_layer[i].data<float>()[j] << " ";
            //     }
            //     out << std::endl;
            // }

    }

    void recv_grad_from_P_to_T_device(int embedding_size, NtsVar &recv_grad)
    {
        cudaSetUsingDevice(global_device_id);
        // LOG_INFO("global device id [%d]", global_device_id);

        NtsVar grad = graph->Nts->NewKeyTensor({owned_vertices, embedding_size}, 
                torch::DeviceType::CUDA, global_device_id);

        ValueType *f_output_buffer=
                graph->Nts->getWritableBuffer(grad, torch::DeviceType::CUDA);

        nccl_pt_comm->P2T_recv_write_to_X_buffer(embedding_size, f_output_buffer);

        recv_grad = grad;
        
            // std::ofstream out("./log/cora_recv_grad_"+ std::to_string(global_device_id) + ".txt", std::ios_base::out);//for debug
            // out << "(out data)GPU ID: " << recv_grad.device() << std::endl;
            // out << "(out data)size: " << recv_grad.sizes() << std::endl;
            // NtsVar X_layer = recv_grad.to(torch::DeviceType::CPU);
            // for(int i = 0; i < owned_vertices; i++)
            // {
            //     out << "dst id[" << i + subdevice_offset[device_id] << "] embedding: ";
            //     for(int j = 0; j < X_layer.size(1); j++)
            //     {
            //         out << X_layer[i].data<float>()[j] << " ";
            //     }
            //     out << std::endl;
            // }

    }


};



//一个GraphChunk是保存水平解耦之后的各个子图
class GraphChunk
{
public:
    Cuda_Stream* cs;
    at::cuda::CUDAStream* ts;

    //用来生成ChunkDep
    VertexId *dstList;
    VertexId *srcList;

    VertexId chunk_id;//chunk的Id
    VertexId chunks;//chunk的数量和

    VertexId gpu_id;
    VertexId gpus;

    VertexId* chunk_offset;
    VertexId owned_vertices;
    VertexId global_vertices;
    VertexId owned_edges;
    
    Graph<Empty> *graph;

    NCCL_Communicator *nccl_comm;

    std::vector<SubgraphSegment*> graph_topo;

    NtsVar feature;
    NtsVar label;
    NtsVar label_gpu;
    NtsVar mask;
    NtsVar mask_gpu;

    std::vector<NtsVar> X_T;
    NtsVar X_TB;
    int K;
    std::vector<NtsVar> X_P;
    std::vector<NtsVar> X_PB;
    NtsVar X_P_APPNP;
    int flag; //第一次SPMM保存

    
    GraphChunk(Graph<Empty> *graph_, VertexId* chunk_offset_, VertexId chunks_, VertexId chunk_id_, VertexId gpus_)
    {
        graph = graph_;
        chunk_offset = chunk_offset_;
        chunks = chunks_;
        chunk_id = chunk_id_;
        gpus = gpus_;

        owned_vertices = chunk_offset[chunk_id+1] - chunk_offset[chunk_id];
        global_vertices = chunk_offset[chunks];
        //边需要图拓扑才能算
    }

    int get_chunk_id(VertexId v_i) {
        for (int i = 0; i < chunks; i++) {
            if (v_i >= chunk_offset[i] && v_i < chunk_offset[i + 1]) {
                return i;
            }
        }
        printf("wrong vertex%d\n", v_i);
        assert(false);
    } 

    void GenerateAll(std::function<ValueType(VertexId, VertexId)> weight_compute, 
                                    VertexId *reorder_column_offset, VertexId *reorder_row_indices)
    {
        owned_edges = reorder_column_offset[chunk_offset[chunk_id+1]] - reorder_column_offset[chunk_offset[chunk_id]];

        printf("-------------------------------------chunk id : %d  \n\t\t  owned edges:%d, owned vertices:%d\n", 
                        chunk_id, owned_edges,owned_vertices);

        generatePartitionedSubgraph(reorder_column_offset, reorder_row_indices);
        std::cout << "finish generate partition subgraphs" << std::endl;
        generateChunkDep(weight_compute);
        std::cout << "finish partition to chunks" << std::endl;
    }

    void generatePartitionedSubgraph(VertexId *reorder_column_offset, VertexId *reorder_row_indices)
    {
        this->dstList = new VertexId[owned_edges];
        this->srcList = new VertexId[owned_edges];
        int write_position=0; 
        for(VertexId local_id = 0; local_id < owned_vertices; local_id++)
        {
            VertexId dst = local_id + chunk_offset[chunk_id];
            for(VertexId index = reorder_column_offset[dst]; index < reorder_column_offset[dst+1]; index++)
            {
                srcList[write_position] = reorder_row_indices[index];
                dstList[write_position++] = dst;
            }
        }

        // std::ofstream out("./log/cora_edgeList" + std::to_string(device_id) + ".txt", std::ios_base::out);
        // for(int i = 0; i < owned_edges; i++)
        // {
        //     out << dstList[i] << " " << srcList[i] << std::endl;
        // }

    }

    void generateChunkDep(std::function<ValueType(VertexId, VertexId)> weight_compute)
    {
        graph_topo.clear();
        std::vector<VertexId>edgecount;
        edgecount.resize(chunks,0);
        std::vector<VertexId>edgenumber;//本地边的src点在对应device的边数量
        edgenumber.resize(chunks,0);

        for(VertexId i=0;i<this->owned_edges;i++){
            VertexId src_partition=get_chunk_id(srcList[i]);
            edgenumber[src_partition]+=1;
        }

        for (VertexId i = 0; i < chunks; i++) {
            graph_topo.push_back(new SubgraphSegment);
            graph_topo[i]->init(chunk_offset[i],
                                chunk_offset[i + 1],
                                chunk_offset[chunk_id],
                                chunk_offset[chunk_id + 1],
                                edgenumber[i]);
            graph_topo[i]->allocVertexAssociateData();
            graph_topo[i]->allocEdgeAssociateData();
        }
        
        for (VertexId i = 0; i < owned_edges; i++) {//设置graph chunk的src list和dst list
            int source = srcList[i];
            int destination = dstList[i];
            int src_partition = get_chunk_id(source);
            int offset = edgecount[src_partition]++;
            graph_topo[src_partition]->source[offset] = source;
            graph_topo[src_partition]->destination[offset] = destination;
        }

        VertexId *tmp_column_offset = new VertexId[global_vertices + 1];
        VertexId *tmp_row_offset = new VertexId[global_vertices + 1];
        for (VertexId i = 0; i < chunks; i++) {
            memset(tmp_column_offset, 0, sizeof(VertexId) * (global_vertices+ 1));
            memset(tmp_row_offset, 0, sizeof(VertexId) * (global_vertices + 1));
            for (VertexId j = 0; j < graph_topo[i]->edge_size; j++) {
                //get offset（local id）
                VertexId v_src_m = graph_topo[i]->source[j];
                VertexId v_dst_m = graph_topo[i]->destination[j];
                VertexId v_dst = v_dst_m - graph_topo[i]->dst_range[0];
                VertexId v_src = v_src_m - graph_topo[i]->src_range[0];

                tmp_column_offset[v_dst + 1] += 1;
                
                tmp_row_offset[v_src + 1] += 1; 
            }

            //calc the partial sum
            graph_topo[i]->column_offset[0] = 0;
            for (VertexId j = 0; j < graph_topo[i]->batch_size_forward; j++) {
                tmp_column_offset[j + 1] += tmp_column_offset[j];
                graph_topo[i]->column_offset[j + 1] = tmp_column_offset[j + 1];
            }
            graph_topo[i]->row_offset[0]=0;
            for (VertexId j = 0; j < graph_topo[i]->batch_size_backward; j++){
                tmp_row_offset[j + 1] += tmp_row_offset[j];
                graph_topo[i]->row_offset[j + 1] = tmp_row_offset[j + 1];
            }

            //calc row indices
            for (VertexId j = 0; j < graph_topo[i]->edge_size; j++) {
                // v_src is from partition i
                // v_dst is from local partition
                VertexId v_src_m = graph_topo[i]->source[j];
                VertexId v_dst_m = graph_topo[i]->destination[j];
                VertexId v_dst = v_dst_m - graph_topo[i]->dst_range[0];
                VertexId v_src = v_src_m - graph_topo[i]->src_range[0];
                // graph_topo[i]->row_indices[tmp_column_offset[v_dst]] = v_src_m;
                //改成局部id，适应spmm
                graph_topo[i]->row_indices[tmp_column_offset[v_dst]] = v_src;
                graph_topo[i]->edge_weight_forward[tmp_column_offset[v_dst]++] =
                    weight_compute(v_src_m, v_dst_m);
                // graph_topo[i]->column_indices[tmp_row_offset[v_src]] = v_dst_m;
                graph_topo[i]->column_indices[tmp_row_offset[v_src]] = v_dst;
                graph_topo[i]->edge_weight_backward[tmp_row_offset[v_src]++] =
                    weight_compute(v_src_m, v_dst_m);
                
            }
            for (VertexId j = 0; j < graph_topo[i]->batch_size_forward; j++) {        
                // save the src and dst in the column format
                VertexId v_dst_m = j+ graph_topo[i]->dst_range[0];
                for(VertexId e_idx=graph_topo[i]->column_offset[j];e_idx<graph_topo[i]->column_offset[j+1];e_idx++){
                    VertexId v_src_m = graph_topo[i]->row_indices[e_idx];
                    graph_topo[i]->source[e_idx] = (long)(v_src_m);
                    graph_topo[i]->destination[e_idx]=(long)(v_dst_m);
                }
            }
        }

        delete[] tmp_column_offset;
        delete[] tmp_row_offset;
    }
    
    void init_cuda_and_comm(NCCL_Communicator * nccl_comm_)
    {
        cudaSetUsingDevice(gpu_id);
        cudaStream_t cuda_stream;
        cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

        cs = new Cuda_Stream();
        cs->setNewStream(cuda_stream);

        ts = new at::cuda::CUDAStream(at::cuda::CUDAStream::UNCHECKED, 
                            at::Stream(at::Stream::UNSAFE, 
                            c10::Device(at::DeviceType::CUDA, gpu_id),
                            reinterpret_cast<int64_t>(cuda_stream)));
        nccl_comm = nccl_comm_;
    }
    
    void load_feat_label_mask(ValueType *reorder_feat, long *reorder_label, int *reorder_mask)
    {
        int feature_size = graph->gnnctx->layer_size[0];
        feature = graph->Nts->NewLeafTensor(reorder_feat + chunk_offset[chunk_id] * feature_size, {owned_vertices, feature_size}, 
                torch::DeviceType::CPU);
        // std::cout << "end load feature" << std::endl;
        
        mask = graph->Nts->NewLeafKIntTensor(reorder_mask + chunk_offset[chunk_id], {owned_vertices, 1});
        // std::cout << "end load mask" << std::endl;
        label = graph->Nts->NewLeafKLongTensor(reorder_label + chunk_offset[chunk_id], {owned_vertices});
        // std::cout
    }

    void load_feat_to_gpu0()
    {
        // LOG_INFO("start load T");
        cudaSetUsingDevice(0);
        for(int i = 0; i < graph->gnnctx->layer_size.size(); i++)
        {
            NtsVar d;
            X_T.push_back(d);
            // LOG_INFO("X_T.size():%d",X_T.size());
        }
        X_T[0] = feature.cuda().set_requires_grad(true);
        // LOG_INFO("X_T.size():%d",X_T.size());
        // LOG_INFO("end load T");
    }

    void load_graph_to_P_gpu(int K_)
    {
        cudaSetUsingDevice(gpu_id);
        label_gpu = label.cuda();
        mask_gpu = mask.cuda();
        K = K_;
        for(int i = 0; i < K + 1; i++)
        {
            NtsVar d;
            X_P.push_back(d);
            X_PB.push_back(d);
        }
        for(int j = 0; j < graph_topo.size(); j++)
        {
            graph_topo[j]->load_graph_tp_per_gpu(cs->stream);
        }
    }
    void free_feat_from_gpu0(){
        cudaSetUsingDevice(0);
        for(int i = 0; i < graph->gnnctx->layer_size.size(); i++)
        {
            X_T.pop_back();
        }
        // X_T[0] = feature.cuda().set_requires_grad(true);
    }
    void free_graph_from_P_gpu(){
        // LOG_INFO("1");
        cudaSetUsingDevice(gpu_id);
        // free(label_gpu.data_ptr());
        // free(mask_gpu.data_ptr());
        label_gpu.resize_(at::IntArrayRef{0});
        mask_gpu.resize_(at::IntArrayRef{0});

        for(int i = 0; i < K + 1; i++)
        {
            X_P.pop_back();
            X_PB.pop_back();
        }
        // LOG_INFO("3");
        for(int j = 0; j < graph_topo.size(); j++)
        {
            graph_topo[j]->free_graph_from_per_gpu(cs->stream);
        }
        // LOG_INFO("4");
    }

    void send_embedding()
    {
        // LOG_INFO("1");
        // CHECK_CUDA_RESULT(cudaDeviceSynchronize());;
        cudaSetUsingDevice(gpu_id);
        // LOG_INFO("gpu_id:%d",gpu_id);
        // LOG_INFO("X_T.size():%d",X_T.size());// 3
        X_P[0] = X_T[X_T.size() - 1].to(at::DeviceType::CUDA, gpu_id);
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        // LOG_INFO("2");
        //x_p_APPNP = SPMM(a * X)

    }

    void SPMM_csc(int layer)
    {   
        // LOG_INFO("11");
        cudaSetUsingDevice(gpu_id);
        // LOG_INFO("12")
        ValueType *input = graph->Nts->getWritableBuffer(X_P[layer], torch::DeviceType::CUDA);
            // LOG_INFO("13");
            NtsVar f_output = graph->Nts->NewKeyTensor({owned_vertices, X_P[layer].size(1)}, 
                                                    torch::DeviceType::CUDA, gpu_id);
            auto sizes1 = f_output.sizes();
            // std::cout<<"f_output:"<<sizes1<<std::endl;
            ValueType *f_output_buffer = graph->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
            // LOG_INFO("12");
            VertexId *col_offset_gpu = graph_topo[chunk_id]->column_offset_gpu;
            VertexId *row_indi_gpu = graph_topo[chunk_id]->row_indices_gpu;
            ValueType *weight_gpu = graph_topo[chunk_id]->edge_weight_forward_gpu;


            VertexId src_start = graph_topo[chunk_id]->src_range[0];
            VertexId src_end = graph_topo[chunk_id]->src_range[1];
            VertexId dst_start = graph_topo[chunk_id]->dst_range[0];
            VertexId dst_end = graph_topo[chunk_id]->dst_range[1];
            
            VertexId forword_batch_size = graph_topo[chunk_id]->batch_size_forward;
            // LOG_INFO("13");
            int edges = graph_topo[chunk_id]->edge_size;
            // LOG_INFO("start spmm layer:%d",layer);
            cs->spmm_csc(input, f_output_buffer, weight_gpu, row_indi_gpu, col_offset_gpu, src_end-src_start, 
                    src_start, src_end, dst_start, dst_end, edges, forword_batch_size, X_P[layer].size(1));
            // LOG_INFO("end spmm layer:%d",layer);

            cs->CUDA_DEVICE_SYNCHRONIZE();
            
            if(graph->config->alpha!=0){
                //APPNP
                // LOG_INFO("enter if");
                if(layer==0){
                    // LOG_INFO("2");
                    X_P_APPNP = f_output*graph->config->alpha;
                    X_P[layer + 1] = f_output;
                    // auto sizes = X_P_APPNP.sizes();
                    // std::cout<<"X_P_APPNP:"<<sizes<<std::endl;
                    // auto sizes1 = f_output.sizes();
                    // std::cout<<"f_output:"<<sizes1<<std::endl;
                    // LOG_INFO("4");
                }
                else{
                    // LOG_INFO("3");
                    X_P[layer + 1] = torch::add(f_output * (1-graph->config->alpha), X_P_APPNP);
                    // X_P[layer + 1]=X_P[layer + 1].add(X_P_APPNP);
                    // LOG_INFO("5");
                }
                
            }
            else{
                // GCN
                X_P[layer + 1] = f_output;
            }
        


    }

    void SPMM_csr(int layer)
    {
        cudaSetUsingDevice(gpu_id);
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        ValueType *input = X_PB[layer].packed_accessor<ValueType, 2>().data();
        X_PB[layer + 1] = torch::zeros_like(X_PB[layer], at::TensorOptions().device_index(gpu_id).requires_grad(false).dtype(
                                                        torch::kFloat));
        
            ValueType *output = X_PB[layer + 1].packed_accessor<ValueType, 2>().data();

            VertexId *row_offset_gpu = graph_topo[chunk_id]->row_offset_gpu;
            VertexId *colum_indices_gpu = graph_topo[chunk_id]->column_indices_gpu;
            ValueType *weight_gpu = graph_topo[chunk_id]->edge_weight_backward_gpu;

            VertexId src_start = graph_topo[chunk_id]->src_range[0];
            VertexId src_end = graph_topo[chunk_id]->src_range[1];
            VertexId dst_start = graph_topo[chunk_id]->dst_range[0];
            VertexId dst_end = graph_topo[chunk_id]->dst_range[1];
            
            VertexId backward_batch_size = graph_topo[chunk_id]->batch_size_backward;

            int edges = graph_topo[chunk_id]->edge_size;

            cs->spmm_csr(input, output, weight_gpu, row_offset_gpu, colum_indices_gpu, dst_end-dst_start, 
                    src_start, src_end, dst_start, dst_end, edges, backward_batch_size, X_P[layer].size(1));
            cs->CUDA_DEVICE_SYNCHRONIZE();
        // CHECK_CUDA_RESULT(cudaDeviceSynchronize());
    }

    void send_grad(int id)
    {
        cudaSetUsingDevice(id);
        
        X_TB = X_PB[K].to(at::DeviceType::CUDA, id);
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());;

        // std::cout << "X_TB size: " << X_TB.sizes() << std::endl;
        // std::cout << "X_TB: " << X_TB.abs().sum() << std::endl;
        // std::cout << "X_TB device: " << X_TB.device() << std::endl << std::endl;
    }

    
    
    void debug_X()
    {
        
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        for(int i = 0; i < X_T.size(); i++)
        {
            std::cout << " X_T[" << i << "] : " ;
            std::cout << "size: " << X_T[i].sizes() << " ";
            std::cout << "sum: " << X_T[i].abs().sum() << " ";
            std::cout << X_T[i].device() << std::endl << std::endl;
        }
        for(int i = 0; i < X_P.size(); i++)
        {
            std::cout << " X_P[" << i << "] : " ;
            std::cout << "size: " << X_P[i].sizes() << " ";
            std::cout << "sum: " << X_P[i].abs().sum() << " ";
            std::cout << X_P[i].device() << std::endl << std::endl;
        }
        for(int i = 0; i < X_PB.size(); i++)
        {
            std::cout << " X_PB[" << i << "] : " ;
            std::cout << "size: " << X_PB[i].sizes() << " ";
            std::cout << "sum: " << X_PB[i].abs().sum() << " ";
            std::cout << X_PB[i].device() << std::endl << std::endl;
        }
        // assert(false);
    }




};






#endif