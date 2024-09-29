#include "core/neutronstar.hpp"

class GCN_Double_Decouple_impl {
public:
    int iterations;

    ValueType learn_rate;
    ValueType weight_decay;
    ValueType drop_rate;
    ValueType alpha;
    ValueType beta1;
    ValueType beta2;
    ValueType epsilon;
    ValueType decay_rate;
    ValueType decay_epoch;

    int num_devices;
    int num_pipelines;
    int K;
    ValueType GCN_alpha;
    
    
    
    
    Graph<Empty> *graph;
    FullyRepGraph* full_rep_graph;

    HoriDecouplePartition *pipeline_partitioner;
    
    
    GNNDatum *gnndatum;
    
    
    

    
    std::vector<Parameter *> P;
    std::vector<std::vector<Parameter *>> device_NN_P;
    std::vector<NtsVar> X;
    std::vector<std::vector<NtsVar>> X_device;
    nts::ctx::NtsContext* ctx;
    std::vector<nts::ctx::NtsContext *> chunk_ctx;
    std::vector<std::vector<VertexId>> layer_per_gpu;

    std::vector<std::vector<nts::ctx::NtsContext *>> device_T_ctx;
    std::vector<std::vector<nts::ctx::NtsContext *>> device_P_ctx;


    
    NtsVar F;
    
    std::vector<ValueType> train_acc;
    std::vector<ValueType> val_acc;
    std::vector<ValueType> test_acc;
    ValueType max_train_acc = 0;
    ValueType max_val_acc = 0;
    ValueType max_test_acc = 0;

    std::vector<unsigned long> train_correct;
    std::vector<unsigned long> val_correct;
    std::vector<unsigned long> test_correct;
    std::vector<double> per_epoch_time;
    double all_epoch_time = 0.0;

    bool resourse_decoupled = false;

    
    std::vector<std::vector<double>> NN_time;
    std::vector<std::vector<double>> g_time;
    std::vector<std::vector<double>> T2P_time;
    std::vector<std::vector<double>> loss_time;
    std::vector<std::vector<double>> gB_time;
    std::vector<std::vector<double>> P2T_time;
    std::vector<std::vector<double>> nnB_time;
    double no_P_time = 0.0;
    
    GCN_Double_Decouple_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false){
        graph = graph_;
        iterations = iterations_;

        num_devices = graph->config->gpu_num;
        if(num_devices == -1) {
            cudaGetDeviceCount(&num_devices);
        }
        assert(num_devices >= 0);
        LOG_INFO("gpu num: %d", num_devices);

        if(graph->config->Decoupled == 1)
        {
            resourse_decoupled = true;
        }
        LOG_INFO("if resourse_decoupled: %d", resourse_decoupled);

        num_pipelines = graph->config->pipeline_num;
        if(num_pipelines <= 0) {
            num_pipelines = 3;
        }
        assert(num_pipelines >= 1);
        LOG_INFO("pipeline num: %d", num_pipelines);

        
        
        graph->init_gnnctx(graph->config->layer_string);
        
        
        
        graph->init_rtminfo();
        graph->rtminfo->process_local = graph->config->process_local;
        graph->rtminfo->reduce_comm = graph->config->process_local;
        graph->rtminfo->copy_data = false;
        graph->rtminfo->process_overlap = graph->config->overlap;
        graph->rtminfo->with_weight = true;
        graph->rtminfo->lock_free = graph->config->lock_free;
        GCN_alpha = graph->config->alpha;
        K = graph->config->K;
    }

    void init_graph(){
        full_rep_graph = new FullyRepGraph(graph);
        full_rep_graph->GenerateAll();
        full_rep_graph->SyncAndLog("read_finish");
        if(num_devices > 1){

        } else {
            ctx=new nts::ctx::NtsContext();
        }
    }

    void init_nn(){
        learn_rate = graph->config->learn_rate;
        weight_decay = graph->config->weight_decay;
        drop_rate = graph->config->drop_rate;
        alpha = graph->config->learn_rate;
        decay_rate = graph->config->decay_rate;
        decay_epoch = graph->config->decay_epoch;
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-9;

        gnndatum = new GNNDatum(graph->gnnctx, graph);
        if (0 == graph->config->feature_file.compare("random")) {
            gnndatum->random_generate();
        } else {
            gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                             graph->config->label_file,
                                             graph->config->mask_file);
        }
        
        

        for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
            P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                        graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                        beta2, epsilon, weight_decay));
        }

        for (int i = 0; i < P.size(); i++) {
            P[i]->init_parameter();
            P[i]->set_decay(decay_rate, decay_epoch);
        }

        
        
        
        


        
    }

    void init_decoupled_version(){
        
        
        pipeline_partitioner = new HoriDecouplePartition(full_rep_graph, gnndatum, num_pipelines, num_devices);
        
        pipeline_partitioner->partition_to_chunk();
        
        pipeline_partitioner->compute_P_T_GPUs(graph->gnnctx->layer_size.size()-1, K);
        
        pipeline_partitioner->alloc_GPU_handle_chunk();
        
        
        
        
        
        
        
        
        for(int i = 0; i < pipeline_partitioner->T_num; i++)
        {
            for(int j = 0; j < P.size(); j++)
            { 
                torch::Device GPU(torch::kCUDA, i);
                P[j]->to(GPU);
                P[j]->Adam_to_GPU();
            }
        }
        
        for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
        {
            
            std::vector<int> arr_(num_devices);
            std::iota(arr_.begin(), arr_.end(), 0);
            NCCL_Communicator *nccl_comm_all = new NCCL_Communicator(num_devices, arr_.data());

            
            pipeline_partitioner->chunks[pip_id]->init_cuda_and_comm(nccl_comm_all);
            if(graph->config->SMALLGRAPH == 1)
            {
                
                pipeline_partitioner->chunks[pip_id]->load_feat_to_gpu0();
                pipeline_partitioner->chunks[pip_id]->load_graph_to_P_gpu(K);
            }
        }
        

        
        
        
        
        
        
        
        
        
    }

    
    NtsVar LOSS(NtsVar &left, NtsVar &label, NtsVar &mask, int pip_id)
    {
        NtsVar predict = left.log_softmax(1);

        NtsVar mask_train = mask.eq(0);
        auto loss_train = torch::nll_loss(
            predict.masked_select(mask_train.expand({mask_train.size(0), predict.size(1)}))
                   .view({-1, predict.size(1)}),
            label.masked_select(mask_train.view({mask_train.size(0)}))
        );
        

        train_correct[pip_id] = getCorrect(0, predict, label, mask);
        val_correct[pip_id] = getCorrect(1, predict, label, mask);
        test_correct[pip_id] = getCorrect(2, predict, label, mask);

        return loss_train;
    }

    long getCorrect(long s, NtsVar &predict, NtsVar &label, NtsVar &mask) { 
        NtsVar mask_train = mask.eq(s);
        NtsVar all_train =
            predict
                .argmax(1)
                .to(torch::kLong)
                .eq(label)
                .to(torch::kLong)
                .masked_select(mask_train.view({mask_train.size(0)}));
        return  all_train.sum(0).item<long>();
    }

    void Update_P()
    {
        for(int i = 0; i < P.size(); i++)
        {
            P[i]->learnG_with_decay_Adam();
            P[i]->next();
        }
        for (int i = 0; i < P.size(); i++) {
           P[i]->zero_grad();
        }
    }
    
    void add_gradient(int pip_id)
    {
        for(int i = 0; i < P.size(); i++)
        {
            
            if(pip_id==0){
               P[i]->reset_gradient(P[i]->W.grad());
            }
            else{
                P[i]->store_gradient(P[i]->W.grad());
            }
            
        }
        
        
        
    }

    void acc(int type, int epoch, bool log = false)
    {
        unsigned correct_num = 0;
        unsigned label_num = 0;
        if(type == 0){
            for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
            {
                correct_num +=  train_correct[pip_id];
                
                if(log){
                    std::cout << "GNNmini::Running.Epoch[" << epoch << "] " << "subgraph:" << pip_id <<":Times["
                        << per_epoch_time[epoch] << "(s)]" << " (nn time:"<< NN_time[pip_id][epoch] 
                        << ",T2P time:"<< T2P_time[pip_id][epoch] << ",g time:"<< g_time[pip_id][epoch] 
                        << ",loss time:"<< loss_time[pip_id][epoch] << ",gB time:"<< gB_time[pip_id][epoch]
                        << ",P2T time:"<< P2T_time[pip_id][epoch] << ",nnB time:"<< nnB_time[pip_id][epoch]
                        << ") "<< std::endl;
                }
            }
            label_num = pipeline_partitioner->train_num;
            ValueType acc = 1.0 * correct_num / label_num;
            max_train_acc = max_train_acc > acc ? max_train_acc : acc;
            if(log){
                LOG_INFO("Train Acc: %f %d %d", acc, correct_num, label_num);
            } else {
                train_acc.push_back(acc);
            }
        } else if(type == 1){
            for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
            {
                correct_num +=  val_correct[pip_id];
            }
            label_num = pipeline_partitioner->val_num;
            ValueType acc = 1.0 * correct_num / label_num;
            max_val_acc = max_val_acc > acc ? max_val_acc : acc;
            if(log){
                LOG_INFO("val Acc: %f %d %d", acc, correct_num, label_num);
            } else {
                val_acc.push_back(acc);
            }
        } else{
            for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
            {
                correct_num +=  test_correct[pip_id];
            }
            label_num = pipeline_partitioner->test_num;
            ValueType acc = 1.0 * correct_num / label_num;
            max_test_acc = max_test_acc > acc ? max_test_acc : acc;
            if(log){
                LOG_INFO("test Acc: %f %d %d", acc, correct_num, label_num);
                test_acc.push_back(acc); 
            } else {
                test_acc.push_back(acc);
            }
        }


    }
    
    NtsVar vertexForward(NtsVar &x) {
        NtsVar y;
        int layer = graph->rtminfo->curr_layer;
        
            y = torch::relu(P[layer]->forward(x)).set_requires_grad(true);
        
        
        
        

        
        return y;
    }
    
    void decoupled_Forward(int pip_id)
    {
        graph->rtminfo->forward = true;

        GraphChunk *graph_chunk = pipeline_partitioner->chunks[pip_id];

        double NN_t = 0.0;
        NN_t -= get_time();
        int layer = 0;

        
        for(int layer = 0; layer < (graph->gnnctx->layer_size.size()-1); layer++)
        {
            graph->rtminfo->curr_layer = layer;
            graph_chunk->X_T[layer + 1] = vertexForward(graph_chunk->X_T[layer]);
        }
        
        NN_t += get_time();
        NN_time[pip_id].push_back(NN_t);
        

        double T2P_t = 0.0;
        T2P_t -= get_time();

        
        graph_chunk->send_embedding();

        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        T2P_t += get_time();
        T2P_time[pip_id].push_back(T2P_t);
        
        double P_t = 0.0;
        P_t -= get_time();
        for(int layer = 0; layer < K; layer++)
        {
            graph_chunk->SPMM_csc(layer);
        }
        
        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        P_t += get_time();
        g_time[pip_id].push_back(P_t);

        
        
        double loss_t = 0.0;
        loss_t -= get_time();
        NtsVar loss = LOSS(graph_chunk->X_P[K], graph_chunk->label_gpu, graph_chunk->mask_gpu, pip_id);
        loss.backward(torch::ones_like(loss), false);

        loss_t += get_time();
        loss_time[pip_id].push_back(loss_t);
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());

        
        double PB_t = 0.0;
        PB_t -= get_time();
        graph_chunk->X_PB[0] = graph_chunk->X_P[K].grad();
        for(int layer = 0; layer < K; layer++)
        {
            graph_chunk->SPMM_csr(layer);
        }
        PB_t += get_time();
        gB_time[pip_id].push_back(PB_t);

        

        
        double P2T_t = 0.0;
        P2T_t -= get_time();
        
        graph_chunk->send_grad(0);

        P2T_t += get_time();
        P2T_time[pip_id].push_back(P2T_t);
        
        
        double nnB_t = 0.0;
        nnB_t -= get_time();
 
        

        graph_chunk->X_T[graph_chunk->X_T.size() - 1].backward(graph_chunk->X_TB, true);
        add_gradient(pip_id);
        nnB_t += get_time();
        nnB_time[pip_id].push_back(nnB_t);        
    }
    
    void run_decoupled_version(){
        train_correct.resize(num_pipelines);
        test_correct.resize(num_pipelines);
        val_correct.resize(num_pipelines);

        NN_time.resize(num_pipelines);
        T2P_time.resize(num_pipelines);
        g_time.resize(num_pipelines);
        loss_time.resize(num_pipelines);
        gB_time.resize(num_pipelines);
        P2T_time.resize(num_pipelines);
        nnB_time.resize(num_pipelines);

        LOG_INFO("GNNmini::[decoupled.GPU.GCNimpl] running [%d] Epoches\n", iterations);
        
        for(int i = 0; i < num_pipelines; i++)
        {
            chunk_ctx.push_back(new nts::ctx::NtsContext());
        }

        for (int i_i = 0; i_i < iterations; i_i++) 
        {
            graph->rtminfo->epoch = i_i;

            for(int i = 0; i < num_pipelines; i++) {
                chunk_ctx[i]->train();
            }

            
            CHECK_CUDA_RESULT(cudaDeviceSynchronize());
            double epoch_time = 0.0;
            epoch_time -= get_time();

            
            
            LOG_INFO("start loop");
            for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
            {
                
                if(graph->config->SMALLGRAPH==0){
                    
                    pipeline_partitioner->chunks[pip_id]->load_feat_to_gpu0();
                    LOG_INFO("pipid:%d, gpuid:%d",pip_id,pipeline_partitioner->chunks[pip_id]->gpu_id);
                    
                    pipeline_partitioner->chunks[pip_id]->load_graph_to_P_gpu(K);
                }
                
                decoupled_Forward(pip_id);
                
                if(graph->config->SMALLGRAPH==0){
                    
                    
                    pipeline_partitioner->chunks[pip_id]->free_feat_from_gpu0();
                    
                    pipeline_partitioner->chunks[pip_id]->free_graph_from_P_gpu();
                }
                
                CHECK_CUDA_RESULT(cudaDeviceSynchronize());

                

            }
            LOG_INFO("end loop");

            CHECK_CUDA_RESULT(cudaDeviceSynchronize());
            epoch_time += get_time();
            per_epoch_time.push_back(epoch_time);

            Update_P();

            acc(0, i_i, true);
            acc(1, i_i, true);
            acc(2, i_i, true);

            

        }

        double all_nn_time = 0.0;
        double all_T2P_time = 0.0;
        double all_g_time = 0.0;
        
        for(int i_i = 1; i_i < iterations; i_i++)
        {
            
            
            
            all_epoch_time += per_epoch_time[i_i];
        }
        std::cout << "--------------------finish algorithm !" << std::endl;
        
        
        
        std::cout << "all_epoch_time: " << all_epoch_time/200 << std::endl;
        std::cout << "max train acc: " << max_train_acc<< std::endl;
        std::cout << "max val acc: " << max_val_acc<< std::endl;
        std::cout << "max test acc: " << max_test_acc<< std::endl;
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        

    }
    
    void run(){
        if(resourse_decoupled)
        {
            LOG_INFO("INIT START");
            init_decoupled_version();
            LOG_INFO("INIT OVER");
            run_decoupled_version();
        }

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    }

    void debug_init_info()
    {
        for(int i = 0; i < full_rep_graph->global_vertices; i++)
        {
            printf("dst[%d]: src[", i);
            for(int j = full_rep_graph->column_offset[i]; j < full_rep_graph->column_offset[i+1]; j++)
            {
                printf("%d ", full_rep_graph->row_indices[j]);
            }
            std::cout << "]" << std::endl;
        }
    }


};