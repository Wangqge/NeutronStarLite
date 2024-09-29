
#ifndef MULTIDEVICEGENERATOR_HPP
#define MICROBATCHGENERATOR_HPP

#include "core/DeviceSubStructure.hpp"
#include "core/metis.hpp"




class MultiDeviceGenerator
{
private:

public:
    FullyRepGraph* whole_graph;
    std::vector<DeviceSubStructure*> subgraph_queq;
    GNNDatum* gnndatum;//包含feature、label、mask等信息
    int threads;

    VertexId *reorder_column_offset;
    VertexId *reorder_row_indices;

    VertexId device_num;
    VertexId* device_offset;//解耦的时候这个变量同时表示graph offset

    ValueType *reorder_feat;
    long *reorder_label;
    int *reorder_mask;

    VertexId train_num;
    VertexId val_num;
    VertexId test_num;

    bool is_CPU = false;

    //资源解耦的时候使用
    std::vector<DeviceSubStructure*> subNN_queq;
    VertexId all_device_num;//这个同理，因为图划分都是用device_num做的，所以解耦的时候device_num代表P的device数量
    // VertexId* subgraph_offset;//不用这个了，直接用上面的device_offset
    VertexId device_subNN_num;
    VertexId* subNN_offset;

    //graph partition
    std::vector<VertexId> vertex_reorder_id; // vertex_reorder_id[原id] = 新id
    std::vector<VertexId> vertex_reorder_reverse_id; // vertex_reorder_id[新id] = 原id
    MetisPartition* metis_partition;
    std::vector<std::vector<int>> partition_result;

    //水平解耦，需要一张删除依赖的子图
    VertexId sub_all_vertex_num;
    VertexId sub_all_edge_num;
    VertexId *subgraph_colum_offset;
    VertexId *subgraph_row_indices;
    ValueType *sub_feat;
    long *sub_label;
    int *sub_mask;

    MultiDeviceGenerator(){}

    MultiDeviceGenerator(FullyRepGraph* whole_graph_, GNNDatum* gnndatum_, VertexId device_num_)
        :whole_graph(whole_graph_), gnndatum(gnndatum_), device_num(device_num_)
    {
        // threads = numa_num_configured_cpus();
        // omp_set_dynamic(0);
        sub_all_vertex_num = whole_graph->global_vertices;
        sub_all_edge_num = whole_graph->global_edges;
        subgraph_colum_offset = whole_graph->column_offset;
        subgraph_row_indices = whole_graph->row_indices;
        sub_feat = gnndatum->local_feature;
        sub_label = gnndatum->local_label;
        sub_mask = gnndatum->local_mask;

    }

    MultiDeviceGenerator(FullyRepGraph* whole_graph_, GNNDatum* gnndatum_, VertexId device_num_, 
            VertexId sub_all_vertex_num_, VertexId sub_all_edge_num_, 
            VertexId *subgraph_colum_offset_, VertexId *subgraph_row_indices_, 
            ValueType *sub_feat_, long *sub_label_, int *sub_mask_)
        :whole_graph(whole_graph_), gnndatum(gnndatum_), device_num(device_num_)
    {
        sub_all_vertex_num = sub_all_vertex_num_;
        sub_all_edge_num = sub_all_edge_num_;

        //直接复制指针，是因为外面传进来的是指针
        subgraph_colum_offset = subgraph_colum_offset_;
        subgraph_row_indices = subgraph_row_indices_;

        //这里复制一下
        sub_feat = new ValueType[sub_all_vertex_num * gnndatum->gnnctx->layer_size[0]];
        sub_label = new long[sub_all_vertex_num];
        sub_mask = new int[sub_all_vertex_num];
        memcpy(sub_feat, sub_feat_, sub_all_vertex_num * sizeof(ValueType) * gnndatum->gnnctx->layer_size[0]);
        memcpy(sub_label, sub_label_, sub_all_vertex_num * sizeof(long));
        memcpy(sub_mask, sub_mask_, sub_all_vertex_num * sizeof(int));
    }

    
    void CPU_version()
    {   std::cout << "CPU Version" << std::endl;

        device_num = 1;
        is_CPU = true;

        device_offset[1] = sub_all_vertex_num;
        reorder_column_offset = subgraph_colum_offset;
        reorder_row_indices = subgraph_row_indices;
        // debug_info();

        std::cout << "start push subgraph queue" << std::endl;

        subgraph_queq.push_back(new DeviceSubStructure(whole_graph->graph_, sub_all_vertex_num, sub_all_edge_num,
                                    device_offset, 0, 0,
                                    sub_all_edge_num,
                                    gnndatum->gnnctx->layer_size[0]));
        
        std::cout << "CPU Version push subgraph queue finished" << std::endl;
        
        subgraph_queq[0]->GenerateAll([&](VertexId src, VertexId dst) {
                    // return nts::op::nts_norm_degree(whole_graph->graph_, src, dst);
                    return 1 / ((ValueType)std::sqrt(reorder_column_offset[dst+1] - reorder_column_offset[dst])
                        * (ValueType)std::sqrt(reorder_column_offset[src+1] - reorder_column_offset[src]));
                    }, reorder_column_offset, reorder_row_indices);

        std::cout << "CPU Version Generate subgraph finished" << std::endl;

        subgraph_queq[0]->load_feature(sub_feat);
        subgraph_queq[0]->load_label_mask(sub_label, sub_mask);
    }
    void read_from_reorder()
     {
     std::cout << "edge file path : " << whole_graph->graph_->filename << std::endl;
     int folder_offset = whole_graph->graph_->filename.find_last_of("//");
     std::string split_path(whole_graph->graph_->filename, 0, folder_offset);
     split_path = split_path + "/split_edge/partition.info";
     std::cout << "split path : " << split_path << std::endl;

     std::ifstream fin;
     fin.open(split_path.c_str(), std::ios::in);
     std::vector<VertexId> vtx;
     if(!fin)
     {
     std::cout << "cannot open file" << std::endl;
     exit(1);
     return;
     }
     std::string line;
    while(std::getline(fin,line))
     {
     int i, vtx_num;
     std::stringstream ss(line);
     ss >> i >> vtx_num;
     std::cout << "vtx num of partition " << i << " " << vtx_num << std::endl;
     vtx.push_back(vtx_num);
     }
    assert(device_num == vtx.size());
     std::cout << "vtx size : " << vtx.size() << std::endl;

    reorder_column_offset = whole_graph->column_offset;
     reorder_row_indices = whole_graph->row_indices;
   reorder_feat = gnndatum->local_feature;
     reorder_label = gnndatum->local_label;
    reorder_mask = gnndatum->local_mask;

     auto train_vertices = get_mask_vertices(0);
     auto val_vertex = get_mask_vertices(1);
     auto test_vertex = get_mask_vertices(2);

     train_num = train_vertices.size();
     val_num = val_vertex.size();
    test_num = test_vertex.size();

     device_offset = new VertexId[vtx.size()+1];
     device_offset[0] = 0;
     for(int i = 0; i < vtx.size(); i++)
     {
     device_offset[i+1] = device_offset[i] + vtx[i];
    }

     std::cout << "partition offset : " << device_offset[0];
     for(int i = 0; i < vtx.size(); i++)
     {
     std::cout << " " << device_offset[i+1];
     }
     std::cout << std::endl;
    }

    void multi_GPU_version()
    {
        // graph_partition();//graph partition and set device_offset
        // reorder_with_partition();
        read_from_reorder();
        // reorder graph tuopu
        // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        // for(int i = 0; i < sub_all_vertex_num; i++)
        // {
        //     printf("dst[%d](pre id: %d ): src[", i, vertex_reorder_reverse_id[i]);
        //     for(int j = reorder_column_offset[i]; j < reorder_column_offset[i+1]; j++)
        //     {
        //         printf("%d (pre id: %d ) ", reorder_row_indices[j], vertex_reorder_reverse_id[reorder_row_indices[j]]);
        //     }
        //     std::cout << "]" << std::endl;
        // }

        for(int i = 0; i < device_num; i++)
        {
            VertexId owned_edges = reorder_column_offset[device_offset[i+1]] - reorder_column_offset[device_offset[i]];
            subgraph_queq.push_back(new DeviceSubStructure(whole_graph->graph_, sub_all_vertex_num, sub_all_edge_num, 
                                        device_offset, device_num, i, //device id
                                        owned_edges, gnndatum->gnnctx->layer_size[0]));
                                        
            subgraph_queq[i]->GenerateAll([&](VertexId src, VertexId dst) {
                            // return nts::op::nts_norm_degree(whole_graph->graph_, vertex_reorder_id[src], vertex_reorder_id[dst]);
                            // return nts::op::nts_norm_degree(whole_graph->graph_, vertex_reorder_reverse_id[src], vertex_reorder_reverse_id[dst]);
                            
                            return 1 / ((ValueType)std::sqrt(reorder_column_offset[dst+1] - reorder_column_offset[dst])
                                * (ValueType)std::sqrt(reorder_column_offset[src+1] - reorder_column_offset[src]));
                                            
                            }, reorder_column_offset, reorder_row_indices);
            
            // std::cout << "finish Generate subgraph : " << i << std::endl;
            
            subgraph_queq[i]->load_feature(reorder_feat);
            subgraph_queq[i]->load_label_mask(reorder_label, reorder_mask);

            std::cout << "finish load feature label mask : " << i << std::endl;
            
        }
    }

    void decoupled_version()
    {
        //1-stage 确定几个GPU做T、几个GPU做P
        printf("GPU num : %d\n", device_num);
        //这里后续需要一个算法来确定数量
        all_device_num = device_num;
        device_subNN_num = 1;
        // device_subNN_num = 2;
        device_num = all_device_num - device_subNN_num;
        printf("NN GPU num : %d\n", device_subNN_num);
        printf("Graph GPU num : %d\n", device_num);

        //默认GPU编号0~device_num-1做T, device_num~all_device_num做P
        assert(device_num > 0);
        assert(device_subNN_num > 0);
        if(device_num > 1)
        {
            graph_partition();//graph partition and set device_offset
            reorder_with_partition();
            
            for(int i = 0; i < device_num; i++)
            {
                VertexId owned_edges = reorder_column_offset[device_offset[i+1]] - reorder_column_offset[device_offset[i]];
                subgraph_queq.push_back(new DeviceSubStructure(whole_graph->graph_, sub_all_vertex_num, sub_all_edge_num, 
                                            device_offset, device_num, i, //device id
                                            owned_edges, gnndatum->gnnctx->layer_size[0], 
                                            0, whole_graph->graph_->config->K+1));
                                            
                subgraph_queq[i]->GenerateAll([&](VertexId src, VertexId dst) {
                            // return nts::op::nts_norm_degree(whole_graph->graph_, vertex_reorder_id[src], vertex_reorder_id[dst]);
                            // return nts::op::nts_norm_degree(whole_graph->graph_, vertex_reorder_reverse_id[src], vertex_reorder_reverse_id[dst]);
                            return 1 / ((ValueType)std::sqrt(reorder_column_offset[dst+1] - reorder_column_offset[dst])
                                * (ValueType)std::sqrt(reorder_column_offset[src+1] - reorder_column_offset[src]));
                            }, reorder_column_offset, reorder_row_indices);
                subgraph_queq[i]->set_PT_Comm_message(all_device_num, device_num);
                
                // subgraph_queq[i]->load_feature_label_mask(reorder_feat, reorder_label, reorder_mask);
                // std::cout << "finish load feature label mask : " << i << std::endl;
                
            }
        }
        else
        {
            //先算三种点的数量
            auto train_vertices = get_mask_vertices(0);
            auto val_vertex = get_mask_vertices(1);
            auto test_vertex = get_mask_vertices(2);
            train_num = train_vertices.size();
            val_num = val_vertex.size();
            test_num = test_vertex.size();

            device_offset = new VertexId[2];
            device_offset[0] = 0;
            device_offset[1] = sub_all_vertex_num;
            reorder_feat = sub_feat;
            reorder_label = sub_label;
            reorder_mask = sub_mask;
            subgraph_queq.push_back(new DeviceSubStructure(whole_graph->graph_, sub_all_vertex_num, sub_all_edge_num,
                                            device_offset, device_num, 0,  
                                            sub_all_edge_num, gnndatum->gnnctx->layer_size[0], 
                                            0, whole_graph->graph_->config->K+1));
            subgraph_queq[0]->GenerateAll([&](VertexId src, VertexId dst) {
                    // return nts::op::nts_norm_degree(whole_graph->graph_, src, dst);
                    return 1 / ((ValueType)std::sqrt(subgraph_colum_offset[dst+1] - subgraph_colum_offset[dst])
                        * (ValueType)std::sqrt(subgraph_colum_offset[src+1] - subgraph_colum_offset[src]));
                    }, subgraph_colum_offset, subgraph_row_indices);
            subgraph_queq[0]->set_PT_Comm_message(all_device_num, device_num);
        }
        // debug_info();

        //NN partition
        int average_vertices = sub_all_vertex_num / device_subNN_num;
        subNN_offset = new VertexId[device_subNN_num + 1];
        subNN_offset[0] = 0;
        for(int i = 0; i < device_subNN_num; i++)
        {
            subNN_offset[i + 1] = subNN_offset[i] + average_vertices;
        }
        if(subNN_offset[device_subNN_num] != sub_all_vertex_num)
        {
            subNN_offset[device_subNN_num] = sub_all_vertex_num;
        }
        std::cout << "NN offset : " << subNN_offset[0];
        for(int i = 0; i < device_subNN_num; i++)
        {
            std::cout << " " << subNN_offset[i + 1];
        }
        std::cout << std::endl;
        for(int i = 0; i < device_subNN_num; i++)
        {
            subNN_queq.push_back(new DeviceSubStructure(whole_graph->graph_, sub_all_vertex_num, sub_all_edge_num, 
                                                        subNN_offset, device_subNN_num,
                                                        i, 0, gnndatum->gnnctx->layer_size[0], device_num));
            subNN_queq[i]->set_PT_Comm_message(all_device_num, device_num);
        }
        std::cout << "finish NN partition" << std::endl;

        generate_PT_message_offset();
        
    }
    
    void generate_PT_message_offset()
    {
        std::cout << "generate PT message offset" << std::endl;
        for(int i = 0; i < device_subNN_num; i++)
        {
            subNN_queq[i]->T2P_offset = new VertexId[device_num + 1];
            subNN_queq[i]->T2P_offset[0] = 0;
            for(int j = 1; j < device_num + 1; j++)
            {
                if(subNN_offset[i] > device_offset[j])
                {
                    subNN_queq[i]->T2P_offset[j] = 0;
                    continue;
                }
                subNN_queq[i]->T2P_offset[j] = (subNN_offset[i+1] < device_offset[j] ? subNN_offset[i+1] : device_offset[j])
                                                -  subNN_offset[i];         
            }
        }
        for(int i = 0; i < device_num; i++)
        {
            subgraph_queq[i]->P2T_offset = new VertexId[device_subNN_num + 1];
            subgraph_queq[i]->P2T_offset[0] = 0;
            for(int j = 1; j < device_subNN_num + 1; j++)
            {
                if(device_offset[i] > subNN_offset[j])
                {
                    subgraph_queq[i]->P2T_offset[j] = 0;
                    continue;
                }
                subgraph_queq[i]->P2T_offset[j] = (device_offset[i+1] < subNN_offset[j] ? device_offset[i+1] : subNN_offset[j])
                                                    -  device_offset[i];
            }
        }
        std::cout << "finished generate PT message offset" << std::endl;

        //debug
        for(int i = 0; i < device_subNN_num; i++)
        {
            std::cout << "subNN_queq[" << i << "]->T2P_offset : ";
            for(int j = 0; j < device_num + 1; j++)
            {
                std::cout << subNN_queq[i]->T2P_offset[j] << " ";
            }
            std::cout << std::endl;
        }
        for(int i = 0; i < device_num; i++)
        {
            std::cout << "subgraph_queq[" << i << "]->P2T_offset : ";
            for(int j = 0; j < device_subNN_num + 1; j++)
            {
                std::cout << subgraph_queq[i]->P2T_offset[j] << " ";
            }
            std::cout << std::endl;
        }
        
    }
    
    void load_data_T_before_P()
    {
        //将原始feature加载到sub_NN_queq，将label和mask加载到sub_graph_queq
        for(int i = 0; i < device_subNN_num; i++)
        {
            subNN_queq[i]->load_feature(reorder_feat);
        }
        for(int i = 0; i < device_num; i++)
        {
            subgraph_queq[i]->load_label_mask(reorder_label, reorder_mask);
        }
    }

    // void init_CudaStream()
    // {
    //     for(int i = 0; i < device_num; i++)
    //     {
    //         subgraph_queq[i]->init_CudaStream();
    //     }
    // }

    void init_communicator(NCCL_Communicator *nccl_comm)
    {
        for(int i = 0; i < subgraph_queq.size(); i++)
        {
            subgraph_queq[i]->init_graph_communicator(nccl_comm);
        }
    }

    void init_communicator_all(NCCL_Communicator *nccl_comm)
    {
        for(int i = 0; i < subgraph_queq.size(); i++)
        {
            subgraph_queq[i]->init_graph_communicator(nccl_comm);
            subgraph_queq[i]->init_PT_communicator(nccl_comm);
        }
        for(int i = 0; i < subNN_queq.size(); i++)
        {
            subNN_queq[i]->init_PT_communicator(nccl_comm);
        }
    }

    void load_data_to_corresponding_device()
    {
        for(int i = 0; i < device_num; i++)
        {
            subgraph_queq[i]->load_data_to_GPUs();
        }
    }

    void load_data_T_before_P_to_corresponding_device()
    {//将原始feature、parameter等加载到sub_NN_queq对应的GPU，将图拓扑加载到sub_graph_queq
        for(int i = 0; i < device_subNN_num; i++)
        {
            subNN_queq[i]->load_NN_data_T_before_P_to_GPU();
        }
        for(int i = 0; i < device_num; i++)
        {
            subgraph_queq[i]->load_graph_data_T_before_P_to_GPU();
        }
    }

    void load_data_T_before_P_to_corresponding_device_without_Para()
    {//将原始feature、parameter等加载到sub_NN_queq对应的GPU，将图拓扑加载到sub_graph_queq
        for(int i = 0; i < device_subNN_num; i++)
        {
            subNN_queq[i]->load_NN_data_T_before_P_to_GPU_without_Para();
        }
        for(int i = 0; i < device_num; i++)
        {
            subgraph_queq[i]->load_graph_data_T_before_P_to_GPU();
        }
    }

    //graph partition and reorder
    std::vector<VertexId> get_mask_vertices(int type)
    {
        std::vector<VertexId> mask_vertices;
        for(int i = 0; i < sub_all_vertex_num; i++)
        {
            if(sub_mask[i] == type)
            {
                mask_vertices.emplace_back(i);
            }
        }
        return mask_vertices;
    }
    void graph_partition()
    {
        assert(device_num>1);
        partition_result.resize(device_num);
        vertex_reorder_id.resize(sub_all_vertex_num);
        vertex_reorder_reverse_id.resize(sub_all_vertex_num);

        device_offset = new VertexId[device_num+1];
        memset(device_offset,0,(device_num)*sizeof(VertexId));

        //get partition with metis
        metis_partition = new MetisPartition(sub_all_vertex_num, device_num);
        metis_partition->SetCSC(subgraph_colum_offset, subgraph_row_indices);

        int metis_dim = whole_graph->graph_->config->metis_dim;
        std::cout << "--------------------graph partition with metis dim: " << metis_dim << std::endl;
        assert(metis_dim < 5);
        assert(metis_dim > 0);
        auto train_vertices = get_mask_vertices(0);
        auto val_vertex = get_mask_vertices(1);
        auto test_vertex = get_mask_vertices(2);

        train_num = train_vertices.size();
        val_num = val_vertex.size();
        test_num = test_vertex.size();

        metis_partition->SetTrainVertices(train_vertices);
        metis_partition->SetValVertices(val_vertex);
        metis_partition->SetTestVertices(test_vertex);
        metis_partition->set_degree();
        // if(metis_dim == 5)//暂时不提供pagerank的
        // {
        //     double *pr = PageRank(graph, 20);
        //     metis_partition->set_pagerange_score(pr);
        // }

        metis_partition->Graph_Partition_With_Multi_Dim_Balance(metis_dim);

        //get parititon result
        const auto& result_ = metis_partition->GetPartitionResult();

        for(int i = 0; i < result_.size(); i++)
        {
            partition_result[result_[i]].push_back(i);
        }

        // for (int i = 0; i < partition_result.size(); i++)
        // {
        //   std::cout << "partition [" << i << "] vertices id :";
        //    for (int j = 0; j < partition_result[i].size(); j++)
        //    {
        //        std::cout << partition_result[i][j] << " ";
        //    }
        //    std::cout << std::endl << "partition[" << i <<"] size : " << partition_result[i].size() << std::endl;
        // }
        // std::cout << std::endl;

        //get vertex reorder id
        int offset = 0;
        for(int i = 0; i < partition_result.size(); i++){
            for (int j = 0; j < partition_result[i].size(); j++)
            {
                vertex_reorder_id[partition_result[i][j]] = offset + j;
                vertex_reorder_reverse_id[offset + j] = partition_result[i][j];
            }
            offset += partition_result[i].size();
            device_offset[i + 1] = offset;
        }

        for(int i = 0; i < device_num+1; i++)
        {
            std::cout << device_offset[i] << " ";
        }
        std::cout << std::endl;
   
        // 输出reorder结果，i：pre id；  vertex_reorder_id[i]: reorder id
        // std::ofstream reorder_out("./log/cora_vertex_reorder.txt", std::ios_base::out);
        // for (int i = 0; i < sub_all_vertex_num; i++)
        // {
        //    reorder_out << i << "==>" << vertex_reorder_id[i] << std::endl;
        // }
        // reorder_out.close();
        
    }
    void reorder_with_partition()
    {
        VertexId* colum_offset = subgraph_colum_offset;
        VertexId* row_indices = subgraph_row_indices;

        reorder_column_offset = new VertexId[sub_all_vertex_num+1];
        reorder_row_indices = new VertexId[sub_all_edge_num];
        reorder_column_offset[0] = 0;

        //first : generate reorder CSC
        //degree
        // #pragma omp parallel for
        for(VertexId new_id = 0; new_id < sub_all_vertex_num; new_id++)
        {
            VertexId old_id = vertex_reorder_reverse_id[new_id];
            reorder_column_offset[new_id+1] = colum_offset[old_id+1] - colum_offset[old_id];
        }

        for(VertexId new_id = 0; new_id < sub_all_vertex_num; new_id++)
        {
            reorder_column_offset[new_id+1] += reorder_column_offset[new_id];
        }
        
        // #pragma omp parallel for
        for(VertexId new_id = 0; new_id < sub_all_vertex_num; new_id++)
        {
            VertexId old_id = vertex_reorder_reverse_id[new_id];
            for(VertexId start = colum_offset[old_id]; start < colum_offset[old_id+1]; start++)
            {
                VertexId old_neighber = row_indices[start];
                VertexId new_neighber = vertex_reorder_id[old_neighber];
                reorder_row_indices[reorder_column_offset[new_id]+start-colum_offset[old_id]] = new_neighber;
            }
        }

        //second : generate reorder label/mask/feature
        reorder_feat = new ValueType[sub_all_vertex_num * gnndatum->gnnctx->layer_size[0]];
        reorder_label = new long[sub_all_vertex_num];
        reorder_mask = new int[sub_all_vertex_num];

        // #pragma omp parallel for
        for(VertexId new_id = 0; new_id < sub_all_vertex_num; new_id++)
        {
            VertexId old_id = vertex_reorder_reverse_id[new_id];
            memcpy(reorder_feat + new_id * gnndatum->gnnctx->layer_size[0], 
                   sub_feat + old_id * gnndatum->gnnctx->layer_size[0],
                   sizeof(ValueType) * gnndatum->gnnctx->layer_size[0]);
            reorder_label[new_id] = sub_label[old_id];
            reorder_mask[new_id] = sub_mask[old_id];
        }
    }



    void debug_info()
    {//debug(输出图划分前和图划分之后以及每个graph chunk的图拓扑)
        std::ofstream before_out("./log/before_part_graph_topo.txt", std::ios_base::out);
        std::ofstream reorder_out("./log/after_part_graph_topo.txt", std::ios_base::out);
        std::ofstream chunk_out("./log/graph_chunk_topo.txt", std::ios_base::out);
        for(int i = 0; i < sub_all_vertex_num; i++)
        {
            before_out << "dst id[" << i << "] src: ";
            for(int j = subgraph_colum_offset[i]; j < subgraph_colum_offset[i+1]; j++)
            {
                before_out << subgraph_row_indices[j] << " ";
            }
            before_out << std::endl;
        }
        if(device_num > 1)
        {
            // for(int i = 0; i < sub_all_vertex_num; i++)
            // {
            //     reorder_out << "dst[" << i << "]:" << ": src: ";
            //     for(int j = reorder_column_offset[i]; j < reorder_column_offset[i+1]; j++)
            //     {
            //         reorder_out << reorder_row_indices[j] << " ";
            //     }
            //     reorder_out << std::endl;
            // }
            // reorder graph tuopu
            for(int i = 0; i < sub_all_vertex_num; i++)
            {
                reorder_out << "dst[" << i << "](pre id:" << vertex_reorder_reverse_id[i] << "): src: ";
                for(int j = reorder_column_offset[i]; j < reorder_column_offset[i+1]; j++)
                {
                    reorder_out << reorder_row_indices[j] << "(pre id:" << vertex_reorder_reverse_id[reorder_row_indices[j]] << ") ";
                }
                reorder_out << std::endl;
            }
        }
        else
        {
            reorder_out << "no reorder!";
        }

        for(int sub = 0; sub < subgraph_queq.size(); sub++)
        {
            chunk_out << "---------------------------------------part[" << sub << "] :" << std::endl;
            for(int chunk = 0; chunk < subgraph_queq[sub]->graph_chunks.size(); chunk++)
            {
                chunk_out << "~~~~~~~~~~~~~~~~~~~graph chunk[" << chunk << "] : " << std::endl;
                for(int i = 0; i < subgraph_queq[sub]->graph_chunks[chunk]->batch_size_forward; i++)
                {
                    chunk_out << "dst[" << i << "]: src: ";
                    for(int j = subgraph_queq[sub]->graph_chunks[chunk]->column_offset[i];
                            j < subgraph_queq[sub]->graph_chunks[chunk]->column_offset[i+1]; j++)
                    {
                        chunk_out << subgraph_queq[sub]->graph_chunks[chunk]->row_indices[j] << " ";
                    }
                    chunk_out << std::endl;
                }
                chunk_out << std::endl;
            }
            chunk_out << std::endl;
        }
        before_out.close();
        reorder_out.close();
        chunk_out.close();
    }

    void debug_partition()
    {
        std::ofstream out("./log/debug_partition.txt", std::ios_base::out);
        for(int i = 0; i < sub_all_vertex_num; i++)
        {
            VertexId reorder_id = vertex_reorder_id[i];
            if((subgraph_colum_offset[i+1] - subgraph_colum_offset[i]) 
                    != reorder_column_offset[reorder_id+1] - reorder_column_offset[reorder_id])
            {
                out << "error! : dst [" << i << "] degree!" << std::endl;
                assert(false);
            }

            int index = 0;
            for(int j = subgraph_colum_offset[i]; j < subgraph_colum_offset[i+1]; j++)
            {
                VertexId reorder_src_id = reorder_row_indices[reorder_column_offset[reorder_id] + index++];
                if(vertex_reorder_reverse_id[reorder_src_id] != subgraph_row_indices[j])
                {
                    out << "error! : src [" << j << "] " << std::endl;
                    assert(false);
                }
            }
            if((i % 500 == 0) && (i != 0))
            {
                out << "finish check reorder topo [" << i << "---" << i+500 << "]"  << std::endl;
            }

            
            //再检测一下reorder feat/mask/label
            for(int j = 0; j < gnndatum->gnnctx->layer_size[0]; j++)
            {
                if(reorder_feat[reorder_id * gnndatum->gnnctx->layer_size[0] + j] != 
                        sub_feat[i * gnndatum->gnnctx->layer_size[0] + j])
                {
                    out << "error! : dst [" << i << "] feat! " << std::endl;
                }
            }

            if(reorder_label[reorder_id] != sub_label[i])
            {
                out << "error! : dst [" << i << "] label! " << std::endl;
            }
            if(reorder_mask[reorder_id] != sub_mask[i])
            {
                out << "error! : dst [" << i << "] mask! " << std::endl;
            }
            if((i % 500 == 0) && (i != 0))
            {
                out << "finish check reorder fea/mask/label [" << i << "---" << i+500 << "]"  << std::endl;
            }
        }

        out << "-----------------------start check graph chuncks feat/mask/label--------------------------" << std::endl;

        for(int sub = 0; sub < subgraph_queq.size(); sub++)
        {
            out << "~~~~~~~~~~~~~~~~~~~~~part[" << sub << "] debug feat/mask/feat: " << std::endl;
            float* subfeat = subgraph_queq[sub]->feature.packed_accessor<ValueType, 2>().data();
            long * sublabel = subgraph_queq[sub]->label.packed_accessor<long, 1>().data();
            int * submask = subgraph_queq[sub]->mask.packed_accessor<int, 2>().data();

            for(int i = 0; i < subgraph_queq[sub]->owned_vertices; i++)
            {
                VertexId id = i + device_offset[sub];
                for(int j = 0; j < gnndatum->gnnctx->layer_size[0]; j++)
                {
                    if(reorder_feat[id * gnndatum->gnnctx->layer_size[0] + j] != subfeat[i * gnndatum->gnnctx->layer_size[0] + j])
                    {                
                        out << "error! : dst [" << i << "] sub feat!" << std::endl;
                        assert(false);
                    }
                }
                if(reorder_label[id] != sublabel[i])
                {
                    out << "error! : dst [" << i << "] sub label!" << std::endl;
                    assert(false);
                }
                if(reorder_mask[id] != submask[i])
                {
                    out << "error! : dst [" << i << "] submask!" << std::endl;
                    assert(false);
                }
                if((i % 500 == 0) && (i != 0))
                {
                    out << "finish check sub feat/mask/label  [" << i << "---" << i+500 << "]"  << std::endl;
                }
            }
        }
        
        out << "-----------------------start check graph chuncks CSC--------------------------" << std::endl;
        for(int sub = 0; sub < subgraph_queq.size(); sub++)
        {
            out << "---------------------------------------part[" << sub << "] :" << std::endl;
            for(int chunk = 0; chunk < subgraph_queq[sub]->graph_chunks.size(); chunk++)
            {
                out << "~~~~~~~~~~~~~~~~~~~graph chunk[" << chunk << "] : " << std::endl;
                for(int i = 0; i < subgraph_queq[sub]->graph_chunks[chunk]->batch_size_forward; i++)
                {
                    out << "dst[" << i + device_offset[sub] << "]: src: ";
                    for(int j = subgraph_queq[sub]->graph_chunks[chunk]->column_offset[i];
                            j < subgraph_queq[sub]->graph_chunks[chunk]->column_offset[i+1]; j++)
                    {
                        out << subgraph_queq[sub]->graph_chunks[chunk]->row_indices[j] << " ";
                    }

                    out << "\t\treorder前id ： dst[" << vertex_reorder_reverse_id[i + device_offset[sub]] << "] src: " ;
                    for(int j = subgraph_colum_offset[vertex_reorder_reverse_id[i + device_offset[sub]]];
                            j < subgraph_colum_offset[vertex_reorder_reverse_id[i + device_offset[sub]]+1]; j++)
                    {
                        out << subgraph_row_indices[j] << " ";
                    }

                    out << std::endl;
                }
                out << std::endl;
            }
            out << std::endl;
        }

        out << "-----------------------start check graph chuncks CSR--------------------------" << std::endl;
        for(int sub = 0; sub < subgraph_queq.size(); sub++)
        {
            out << "---------------------------------------part[" << sub << "] :" << std::endl;
            for(int chunk = 0; chunk < subgraph_queq[sub]->graph_chunks.size(); chunk++)
            {
                out << "~~~~~~~~~~~~~~~~~~~graph chunk[" << chunk << "] : " << std::endl;
                for(int i = 0; i < subgraph_queq[sub]->graph_chunks[chunk]->batch_size_forward; i++)
                {
                    out << "src[" << i + device_offset[chunk] << "]: dst: ";
                    for(int j = subgraph_queq[sub]->graph_chunks[chunk]->row_offset[i];
                            j < subgraph_queq[sub]->graph_chunks[chunk]->row_offset[i+1]; j++)
                    {
                        out << subgraph_queq[sub]->graph_chunks[chunk]->column_indices[j] << " ";
                    }

                    out << std::endl;
                }
                out << std::endl;
            }
            out << std::endl;
        }
        out.close();
    }


};

class HoriDecouplePartition
{
public:
    std::vector<MultiDeviceGenerator*> multi_device_generator_queq;//每一个值是一个子图，用于流水线处理

    VertexId *reorder_column_offset;
    VertexId *reorder_row_indices;

    ValueType *reorder_feat;
    long *reorder_label;
    int *reorder_mask;

    std::vector<std::vector<VertexId>> delete_dep_column_offset;//用于保存删除了子图之间依赖CSC
    std::vector<std::vector<VertexId>> delete_dep_row_indices;//注意，这里的row_indices用的是local id，为了之后子图训练方便

    std::vector<std::vector<VertexId>> dep_column_offset;//用于保存被删除的子图之间的依赖CSC
    std::vector<std::vector<VertexId>> dep_row_indices;

    int pipline_num;//划分的分区数
    int num_devices;

    MultiDeviceGenerator* graph_partition;//用于划分子图
    VertexId *partition_offset;//每个子图的起始点

    FullyRepGraph* whole_graph;
    GNNDatum* gnndatum;//包含feature、label、mask等信息

    std::vector<GraphChunk*> chunks;

    int T_num;
    int P_num;

    VertexId train_num;
    VertexId test_num;
    VertexId val_num;

    HoriDecouplePartition(FullyRepGraph* whole_graph_, GNNDatum* gnndatum_, VertexId pipeline_num_, VertexId num_devices_)
        :whole_graph(whole_graph_), gnndatum(gnndatum_), pipline_num(pipeline_num_), num_devices(num_devices_)
    {}

    std::vector<VertexId> get_mask_vertices(int type)
    {
        std::vector<VertexId> mask_vertices;
        for(int i = 0; i < whole_graph->global_vertices; i++)
        {
            if(gnndatum->local_mask[i] == type)
            {
                mask_vertices.emplace_back(i);
            }
        }
        return mask_vertices;
    }

    void delete_partition_dep()
    {

        partition_and_reorder();

        delete_dep_column_offset.resize(pipline_num);
        delete_dep_row_indices.resize(pipline_num);
        dep_column_offset.resize(pipline_num);
        dep_row_indices.resize(pipline_num);
        for(int i = 0; i < pipline_num; i++)
        {
            //首先生成每个子图删除跨分区依赖之后的 multi_device_generator_queq
            VertexId subgraph_vertice_num = partition_offset[i+1] - partition_offset[i];
            delete_dep_column_offset[i].push_back(0);
            dep_column_offset[i].push_back(0);
            for(int v = 0; v < subgraph_vertice_num; v++)
            {
                VertexId dst_id = partition_offset[i] + v;
                int delete_dst_degree = 0;
                int dep_dst_degree = 0;
                for(int j = reorder_column_offset[dst_id]; j < reorder_column_offset[dst_id + 1]; j++)
                {
                    VertexId src_id = reorder_row_indices[j];
                    if((src_id >= partition_offset[i])&&(src_id < partition_offset[i + 1]))
                    {
                        delete_dep_row_indices[i].push_back(src_id - partition_offset[i]);
                        delete_dst_degree++;
                    }
                    else
                    {
                        dep_row_indices[i].push_back(src_id);
                        dep_dst_degree++;
                    }
                }
                delete_dep_column_offset[i].push_back(delete_dst_degree + delete_dep_column_offset[i][v]);
                dep_column_offset[i].push_back(dep_dst_degree + dep_column_offset[i][v]);
            }
        }
        
        // debug_info();
        // assert(false);
    }

    void generate_multiDevice_decouple_version()
    {
        for(int i = 0; i < pipline_num; i++)
        {
            multi_device_generator_queq.push_back(new MultiDeviceGenerator(whole_graph, gnndatum, num_devices, 
                delete_dep_column_offset[i].size() - 1, delete_dep_row_indices[i].size(),
                delete_dep_column_offset[i].data(), delete_dep_row_indices[i].data(),
                graph_partition->reorder_feat + partition_offset[i] * gnndatum->gnnctx->layer_size[0],
                graph_partition->reorder_label + partition_offset[i],
                graph_partition->reorder_mask + partition_offset[i] )
                );
        }
        
        // debug_multi_device_generator_queq();
        // assert(false);
    }

    void partition_and_reorder()
    {
        //先根据pipeline数量做一个图划分，然后reorder
        if(pipline_num > 1)
        {        
            graph_partition = new MultiDeviceGenerator(whole_graph, gnndatum, pipline_num);
            graph_partition->graph_partition();
            graph_partition->reorder_with_partition();
            // graph_partition->debug_info();

            reorder_column_offset = graph_partition->reorder_column_offset;
            reorder_row_indices = graph_partition->reorder_row_indices;
            partition_offset = graph_partition->device_offset;
        } 
        else{
            reorder_column_offset = whole_graph->column_offset;
            reorder_row_indices = whole_graph->row_indices;
            graph_partition = new MultiDeviceGenerator(whole_graph, gnndatum, pipline_num);
            graph_partition->reorder_feat = gnndatum->local_feature;
            graph_partition->reorder_label = gnndatum->local_label;
            graph_partition->reorder_mask = gnndatum->local_mask;
            partition_offset = new VertexId[2]; //
            partition_offset[0] = 0;
            partition_offset[1] = whole_graph->global_vertices;
        }
    }

    void read_from_reorder()
    {
        std::cout << "edge file path : " << whole_graph->graph_->filename << std::endl;
        int folder_offset = whole_graph->graph_->filename.find_last_of("//");
        std::string split_path(whole_graph->graph_->filename, 0, folder_offset);
        split_path = split_path + "/split_edge/partition.info";
        std::cout << "split path : " << split_path << std::endl;

        std::ifstream fin;
        fin.open(split_path.c_str(), std::ios::in);
        std::vector<VertexId> vtx;
        if (!fin)
        {
            std::cout << "cannot open file" << std::endl;
            exit(1);
            return;
        }
        std::string line;
        while(std::getline(fin, line))
        {
            int i, vtx_num;
            std::stringstream ss(line);
            ss >> i >> vtx_num;
            std::cout << " lalalal " << i << " " << vtx_num << std::endl;
            vtx.push_back(vtx_num);
        }
        assert(pipline_num == vtx.size());
        std::cout << "vtx size : " << vtx.size() << std::endl;


        reorder_column_offset = whole_graph->column_offset;
        reorder_row_indices = whole_graph->row_indices;
        graph_partition = new MultiDeviceGenerator(whole_graph, gnndatum, pipline_num);
        graph_partition->reorder_feat = gnndatum->local_feature;
        graph_partition->reorder_label = gnndatum->local_label;
        graph_partition->reorder_mask = gnndatum->local_mask;
        // partition_offset = new VertexId[2];
        // partition_offset[0] = 0;
        // partition_offset[1] = whole_graph->global_vertices;

        partition_offset = new VertexId[vtx.size() + 1];
        partition_offset[0] = 0;
        for(int i = 0; i < vtx.size(); i++)
        {
            partition_offset[i+1] = partition_offset[i] + vtx[i];
        }

        std::cout << "partition offset : " << partition_offset[0];
        for(int i = 0; i < vtx.size(); i++)
        {
            std::cout << " " << partition_offset[i+1];
        }
        std::cout << std::endl;

    }
    
    void partition_to_chunk()//与delete_partition_dep不同，这里删掉边之后将删掉的边留下来，供日后使用
    {
            // LOG_INFO("1");
            partition_and_reorder(); //这里注释

        


        reorder_feat = graph_partition->reorder_feat;
        reorder_label = graph_partition->reorder_label;
        reorder_mask = graph_partition->reorder_mask;

        auto train_vertices = get_mask_vertices(0);
        auto val_vertex = get_mask_vertices(1);
        auto test_vertex = get_mask_vertices(2);
        train_num = train_vertices.size();
        val_num = val_vertex.size();
        test_num = test_vertex.size();

        for(int i = 0; i < pipline_num; i++)
        {
            chunks.push_back(new GraphChunk(whole_graph->graph_, partition_offset, pipline_num, i, num_devices));
            chunks[i]->GenerateAll([&](VertexId src, VertexId dst) {
                            if(whole_graph->graph_->config->alpha!=0){
                                // APPNP
                                // LOG_INFO("Running APPNP");
                                return (1-whole_graph->graph_->config->alpha) / ((ValueType)std::sqrt(reorder_column_offset[dst+1] - reorder_column_offset[dst])
                                * (ValueType)std::sqrt(reorder_column_offset[src+1] - reorder_column_offset[src]));
                            }
                            // GCN
                            // LOG_INFO("Running GCN");
                            return 1 / ((ValueType)std::sqrt(reorder_column_offset[dst+1] - reorder_column_offset[dst])
                                * (ValueType)std::sqrt(reorder_column_offset[src+1] - reorder_column_offset[src]));
                            }, //APPNP
                reorder_column_offset, reorder_row_indices);
            
            chunks[i]->load_feat_label_mask(reorder_feat, reorder_label, reorder_mask);
        }

        // debug_chunk();

    }

    void compute_P_T_GPUs(int layer_T, int layer_P)
    {
        T_num = 1;
        P_num = num_devices - T_num;

    }

    void alloc_GPU_handle_chunk()
    {
        //分配每个graph chunk的GPU id
        for(int i = 0; i < chunks.size(); i++)
        {
            chunks[i]->gpu_id =  i % P_num + T_num;

            // std::cout << "chunk id : " << chunks[i]->chunk_id << " gpu id: " << chunks[i]->gpu_id << std::endl;
        }
    }
    
    
    void debug_chunk()
    {
        std::ofstream out("./log/debug_chunk.txt", std::ios_base::out);
        out << "-----------------------start check graph chuncks CSC--------------------------" << std::endl;
        for(int sub = 0; sub < chunks.size(); sub++)
        {
            out << "---------------------------------------part[" << sub << "] :" << std::endl;
            for(int chunk = 0; chunk < chunks[sub]->graph_topo.size(); chunk++)
            {
                out << "~~~~~~~~~~~~~~~~~~~graph Topo[" << chunk << "] : " << std::endl;
                for(int i = 0; i < chunks[sub]->graph_topo[chunk]->batch_size_forward; i++)
                {
                    out << "dst[" << i + partition_offset[sub] << "]: src: ";
                    for(int j = chunks[sub]->graph_topo[chunk]->column_offset[i];
                            j < chunks[sub]->graph_topo[chunk]->column_offset[i+1]; j++)
                    {
                        out << chunks[sub]->graph_topo[chunk]->row_indices[j] << " ";
                        out << chunks[sub]->graph_topo[chunk]->edge_weight_forward[j] << " ";
                    }

                    // out << "\t\t reorder前id ： dst[" << graph_partition->vertex_reorder_reverse_id[i + partition_offset[sub]] << "] src: " ;
                    // for(int j = whole_graph->column_offset[graph_partition->vertex_reorder_reverse_id[i + partition_offset[sub]]];
                    //         j < whole_graph->column_offset[graph_partition->vertex_reorder_reverse_id[i + partition_offset[sub]]+1]; j++)
                    // {
                    //     out << whole_graph->row_indices[j] << " ";
                    // }

                    // out << "\t\t reorder后id ： dst[" << i + partition_offset[sub] << "] src: " ;
                    // for(int j = reorder_column_offset[i + partition_offset[sub]];
                    //         j < reorder_column_offset[i + partition_offset[sub]+1]; j++)
                    // {
                    //     out << reorder_row_indices[j] << " ";
                    // }

                    out << std::endl;
                }
                out << std::endl;
            }
            out << std::endl;
        }

        out << "-----------------------start check graph chuncks CSR--------------------------" << std::endl;
        for(int sub = 0; sub < chunks.size(); sub++)
        {
            out << "---------------------------------------part[" << sub << "] :" << std::endl;
            for(int chunk = 0; chunk < chunks[sub]->graph_topo.size(); chunk++)
            {
                out << "~~~~~~~~~~~~~~~~~~~graph chunk[" << chunk << "] : " << std::endl;
                for(int i = 0; i < chunks[sub]->graph_topo[chunk]->batch_size_backward; i++)
                {
                    out << "src[" << i + partition_offset[chunk] << "]: dst: ";
                    for(int j = chunks[sub]->graph_topo[chunk]->row_offset[i];
                            j < chunks[sub]->graph_topo[chunk]->row_offset[i+1]; j++)
                    {
                        out << chunks[sub]->graph_topo[chunk]->column_indices[j] << " ";
                        out << chunks[sub]->graph_topo[chunk]->edge_weight_backward[j] << " ";
                    }

                    out << std::endl;
                }
                out << std::endl;
            }
            out << std::endl;
        }
        out.close();
    }

    void debug_info()
    {
        std::ofstream delete_dep_topo_out("./log/delete_dep_topo.txt", std::ios_base::out);
        std::ofstream dep_topo_out("./log/dep_topo.txt", std::ios_base::out);
        std::ofstream xinxi_out("./log/reorder_xinxi.txt", std::ios_base::out);
        for(int i = 0; i < pipline_num; i++)
        {
            delete_dep_topo_out << "-----------------------subgraph : " << i << std::endl;
            dep_topo_out << "-----------------------subgraph : " << i << std::endl;
            VertexId subgraph_vertice_num = partition_offset[i+1] - partition_offset[i];
            for(int v = 0; v < subgraph_vertice_num; v++)
            {
                VertexId dst_id = partition_offset[i] + v;
                delete_dep_topo_out << "dst [" << dst_id << "] local src :";
                for(int j = delete_dep_column_offset[i][v]; j < delete_dep_column_offset[i][v + 1]; j++)
                {
                    VertexId src_id = delete_dep_row_indices[i][j] + partition_offset[i];
                    delete_dep_topo_out << src_id << " ";
                }
                delete_dep_topo_out << std::endl;

                dep_topo_out << "dst [" << dst_id << "] local src :";
                for(int j = dep_column_offset[i][v]; j < dep_column_offset[i][v + 1]; j++)
                {
                    VertexId src_id = dep_row_indices[i][j];
                    dep_topo_out << src_id << " ";
                }
                dep_topo_out << std::endl;
            }
        }

        for(int i = 0; i < whole_graph->global_vertices; i++)
        {
            xinxi_out << i << " " << graph_partition->reorder_label[i] << "(label) " << graph_partition->reorder_mask[i] << "(mask) ";
            for(int j = 0; j < gnndatum->gnnctx->layer_size[0]; j++)
            {
                xinxi_out << graph_partition->reorder_feat[i * gnndatum->gnnctx->layer_size[0] + j] << " ";
            }
            xinxi_out << std::endl;
        }
    }

    void debug_multi_device_generator_queq()
    {
        // std::ofstream xinxi_out("./log/multi_device_generator_queq_xinxi.txt", std::ios_base::out);
        for(int i = 0; i < pipline_num; i++)
        {
            // xinxi_out << "----------------subgraph " << i << std::endl;
            for(int v = 0; v < delete_dep_column_offset[i].size() - 1; v++)
            {
                VertexId dst_id = v + partition_offset[i];
                assert(multi_device_generator_queq[i]->sub_label[v] == graph_partition->reorder_label[dst_id]);
                assert(multi_device_generator_queq[i]->sub_mask[v] == graph_partition->reorder_mask[dst_id]);
                for(int k = 0; k < gnndatum->gnnctx->layer_size[0]; k++)
                {
                    int index1 = v * gnndatum->gnnctx->layer_size[0] + k;
                    int index2 = dst_id * gnndatum->gnnctx->layer_size[0] + k;
                    assert(multi_device_generator_queq[i]->sub_feat[index1] == graph_partition->reorder_feat[index2]);
                }
                // xinxi_out << dst_id << " " << multi_device_generator_queq[i]->sub_label[v] << "(label) " 
                //             << multi_device_generator_queq[i]->sub_mask[v] << "(mask) ";
                // for(int j = 0; j < gnndatum->gnnctx->layer_size[0]; j++)
                // {
                //     xinxi_out << multi_device_generator_queq[i]->sub_feat[v * gnndatum->gnnctx->layer_size[0] + j] << " ";
                // }
                // xinxi_out << std::endl;
            }
        }
    }


};










#endif