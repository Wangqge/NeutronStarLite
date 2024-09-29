#include "core/neutronstar.hpp"

class APPNP_impl {
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
    ValueType appnp_alpha;
    
    
    
    
    Graph<Empty> *graph;
    FullyRepGraph* full_rep_graph;

    MultiDeviceGenerator* subgraph_generator; 
    
    
    GNNDatum *gnndatum;
    
    
    

    
    std::vector<Parameter *> P;
    std::vector<NtsVar> X;
    nts::ctx::NtsContext* ctx;
    std::vector<nts::ctx::NtsContext *> device_ctx;

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

    
    std::vector<double> NN_time;
    std::vector<double> g_time;
    std::vector<double> T2P_time;
    double no_P_time = 0.0;


    
    APPNP_impl(Graph<Empty> *graph_, int iterations_,
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
        num_pipelines = 1;
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
        appnp_alpha = graph->config->alpha;
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
        train_correct.resize(num_pipelines, 0);
        val_correct.resize(num_pipelines, 0);
        test_correct.resize(num_pipelines, 0);

        
        
        subgraph_generator = new MultiDeviceGenerator(full_rep_graph, gnndatum, num_devices);
        subgraph_generator->decoupled_version();
        subgraph_generator->load_data_T_before_P();
        

        
        std::vector<int> arr_(num_devices);
        std::iota(arr_.begin(), arr_.end(), 0);
        NCCL_Communicator *nccl_comm_all = new NCCL_Communicator(num_devices, arr_.data());

        
        std::vector<int> arr(subgraph_generator->device_subNN_num);
        std::iota(arr.begin(), arr.end(), subgraph_generator->device_num);
        NCCL_Communicator *nccl_comm_NN = new NCCL_Communicator(subgraph_generator->device_subNN_num, arr.data(), subgraph_generator->device_num);
        
        for(int i = 0; i < subgraph_generator->device_subNN_num; i++) 
        {
            for(int j = 0; j < graph->gnnctx->layer_size.size() - 1; j++) {
                subgraph_generator->subNN_queq[i]->P.push_back(new Parameter(graph->gnnctx->layer_size[j], graph->gnnctx->layer_size[j + 1], 
                                                                                alpha, beta1, beta2, epsilon, weight_decay));
                subgraph_generator->subNN_queq[i]->P[j]->W.set_data(P[j]->W);
                subgraph_generator->subNN_queq[i]->P[j]->set_decay(decay_rate, decay_epoch);

                subgraph_generator->subNN_queq[i]->P[j]->set_multi_gpu_comm(nccl_comm_NN);
            }
        }
        

        subgraph_generator->load_data_T_before_P_to_corresponding_device();
        
        subgraph_generator->init_communicator_all(nccl_comm_all);

    }
    
    
    NtsVar LOSS(NtsVar &left, NtsVar &label, nts::ctx::NtsContext* ctx, NtsVar &mask, int pip_id)
    {
        NtsVar predict = left.log_softmax(1);

        NtsVar mask_train = mask.eq(0);
        auto loss_train = torch::nll_loss(
            predict.masked_select(mask_train.expand({mask_train.size(0), predict.size(1)}))
                   .view({-1, predict.size(1)}),
            label.masked_select(mask_train.view({mask_train.size(0)}))
        );
        if(ctx->training == true)
        {
            ctx->appendNNOp(left, loss_train);
        }
        LOG_INFO("loss:%f", loss_train.item<float>());

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

    void Update(DeviceSubStructure *subgraph)
    {
        for(int i = 0; i < subgraph->P.size(); i++)
        {
            subgraph->P[i]->reduce_multi_gpu_gradient(subgraph->P[i]->W.grad(), subgraph->device_id, subgraph->cs->stream);
            subgraph->P[i]->learnG_with_decay_Adam();
            
            subgraph->P[i]->next();
        }
    }
    
    void acc(int type, int epoch, bool log = false)
    {
        unsigned correct_num = 0;
        if(type == 0){
            for(auto&& num : train_correct)
            {
                correct_num += num;
            }
            ValueType acc = 1.0 * correct_num / subgraph_generator->train_num;
            max_train_acc = max_train_acc > acc ? max_train_acc : acc;
            if(log){
                std::cout << "GNNmini::Running.Epoch[" << epoch << "]:Times["
                    << per_epoch_time[epoch] << "(s)]" << " (nn time:"<< NN_time[epoch] 
                    << ",T2P time:"<< T2P_time[epoch] << ",g time:"<< g_time[epoch] << ") "<< std::endl;
                LOG_INFO("Train Acc: %f %d %d", acc, correct_num, subgraph_generator->train_num);
            } else {
                train_acc.push_back(acc);
            }
        } else if(type == 1){
            for(auto&& num : val_correct)
            {
                correct_num += num;
            }
            ValueType acc = 1.0 * correct_num / subgraph_generator->val_num;
            max_val_acc = max_val_acc > acc ? max_val_acc : acc;
            if(log){
                LOG_INFO("val Acc: %f %d %d", acc, correct_num, subgraph_generator->val_num);
            } else {
                val_acc.push_back(acc);
            }
        } else{
            for(auto&& num : test_correct)
            {
                correct_num += num;
            }
            ValueType acc = 1.0 * correct_num / subgraph_generator->test_num;
            max_test_acc = max_test_acc > acc ? max_test_acc : acc;
            if(log){
                LOG_INFO("test Acc: %f %d %d", acc, correct_num, subgraph_generator->test_num);
            } else {
                test_acc.push_back(acc);
            }
        }


    }
    
    void decoupled_T(DeviceSubStructure *subNN, int pip_id)
    {
        

        int dev_id = subNN->global_device_id;
        int local_dev_id = subNN->device_id;
        at::cuda::setCurrentCUDAStream(*(subNN->ts));
        cudaSetUsingDevice(dev_id);
        for(int layer = 0; layer < (graph->gnnctx->layer_size.size()-1); layer++)
        {
            subNN->X[layer + 1] = device_T_ctx[pip_id][local_dev_id]->runVertexForward([&](NtsVar n_i, NtsVar v_i){
                NtsVar y;
                    auto b = subNN->P[layer]->forward(n_i);
                    y = torch::dropout(torch::relu(b), drop_rate, device_T_ctx[pip_id][local_dev_id]->is_train());
                return y;
            }, subNN->X[layer], subNN->X[layer]);

                
                
                
                
                
                
                
                
                
                
                
                
                

        }
        subNN->cs->CUDA_DEVICE_SYNCHRONIZE();

        
        

    }
    
    void decoupled_P(DeviceSubStructure *subgraph, int embedding_size, int pip_id)
    {
        
        

        
        
            
            
            
            
            
            
            
            
            
            
            
            
            

        int dev_id = subgraph->global_device_id;
        int local_dev_id = subgraph->device_id;
        cudaSetUsingDevice(dev_id);
        
        for(int layer = 0; layer < graph->config->K; layer++)
        {
            subgraph->X[layer + 1] = device_P_ctx[pip_id][local_dev_id]->runGraphOp<nts::op::MultiGPUAllGNNCalcGraphSumOp>
                                        (subgraph, subgraph->X[layer]);

                
                
                
                
                
                
                
                
                
                
                
                
                
        }
        
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());

        no_P_time = 0.0;
        no_P_time -= get_time();
        NtsVar loss = LOSS(subgraph->X[graph->config->K], subgraph->label_gpu, device_P_ctx[pip_id][local_dev_id], 
            subgraph->mask_gpu, pip_id);


        device_P_ctx[pip_id][local_dev_id]->self_decouple_P_backward(subgraph, false);
        
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        no_P_time += get_time();
    }

    void decoupled_Forward(int pip_id)
    {
        graph->rtminfo->forward = true;
        
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        double NN_t = 0.0;
        NN_t -= get_time();

        
        std::vector<std::thread> NN_device_forward;
        for(int dev = 0; dev < subgraph_generator->device_subNN_num; dev++)
        {
            NN_device_forward.emplace_back([&](int dev_){ 
                assert(dev_ == subgraph_generator->subNN_queq[dev_]->device_id);                   
                decoupled_T(subgraph_generator->subNN_queq[dev_], pip_id);
            }, dev);
        }
        for(int dev = 0; dev < subgraph_generator->device_subNN_num; dev++){
            NN_device_forward[dev].join();
        }
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        NN_t += get_time();
        NN_time.push_back(NN_t);


        double T2P_t = 0.0;
        T2P_t -= get_time();

        
        int embedding_size = subgraph_generator->subNN_queq[0]->X[subgraph_generator->subNN_queq[0]->X.size() - 1].size(1);
        std::vector<std::thread> transfer_thread;
        for(int dev = 0; dev < subgraph_generator->device_subNN_num; dev++)
        {
            transfer_thread.emplace_back([&](int dev_){ 
                subgraph_generator->subNN_queq[dev_]->send_emb_from_T_to_P_device();
            }, dev);
        }
        for(int dev = 0; dev < subgraph_generator->device_num; dev++)
        {
            transfer_thread.emplace_back([&](int dev_){
                subgraph_generator->subgraph_queq[dev_]->recv_emb_from_T_to_P_device(embedding_size);
            }, dev);
        }
        for(int dev = 0; dev < subgraph_generator->all_device_num; dev++)
        {
            transfer_thread[dev].join();
        }
        
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        T2P_t += get_time();
        T2P_time.push_back(T2P_t);

        double P_t = 0.0;
        P_t -= get_time();

        
        std::vector<std::thread> graph_device_forward;
        for(int dev = 0; dev < subgraph_generator->device_num; dev++)
        {
            graph_device_forward.emplace_back([&](int dev_){
                assert(dev_ == subgraph_generator->subgraph_queq[dev_]->device_id);
                decoupled_P(subgraph_generator->subgraph_queq[dev_], embedding_size, pip_id);
            }, dev);
        }
        for(int dev = 0; dev < subgraph_generator->device_num; dev++)
        {
            graph_device_forward[dev].join();
        }
        CHECK_CUDA_RESULT(cudaDeviceSynchronize());

        P_t += get_time();
        g_time.push_back(P_t - no_P_time);

        
        
        


        
        int grad_size = subgraph_generator->subgraph_queq[0]->decoupled_mid_grad.size(1);
        std::vector<std::thread> transfer_grad_thread;
        for(int dev = 0; dev < subgraph_generator->device_subNN_num; dev++)
        {
            transfer_grad_thread.emplace_back([&](int dev_){ 
                auto &recv_grad = device_T_ctx[pip_id][dev_]->output_grad[device_T_ctx[pip_id][dev_]->top_idx()];
                subgraph_generator->subNN_queq[dev_]->recv_grad_from_P_to_T_device(grad_size, recv_grad);
            }, dev);
        }
        for(int dev = 0; dev < subgraph_generator->device_num; dev++)
        {
            transfer_grad_thread.emplace_back([&](int dev_){
                subgraph_generator->subgraph_queq[dev_]->send_grad_from_P_to_T_device();
            }, dev);
        }
        for(int dev = 0; dev < subgraph_generator->all_device_num; dev++)
        {
            transfer_grad_thread[dev].join();
        }

        CHECK_CUDA_RESULT(cudaDeviceSynchronize());

        
        std::vector<std::thread> NN_device_backward;
        for(int dev = 0; dev < subgraph_generator->device_subNN_num; dev++)
        {
            NN_device_backward.emplace_back([&](int dev_){ 
                assert(dev_ == subgraph_generator->subNN_queq[dev_]->device_id);      
                
                
                
                device_T_ctx[pip_id][dev_]->self_decouple_T_backward(false);
                
                Update(subgraph_generator->subNN_queq[dev_]);
            }, dev);
        }
        for(int dev = 0; dev < subgraph_generator->device_subNN_num; dev++){
            NN_device_backward[dev].join();
        }
    
    }
    
    void run_decoupled_version(){
        LOG_INFO("GNNmini::[decoupled.GPU.APPNPimpl] running [%d] Epoches\n", iterations);
        device_T_ctx.resize(num_pipelines);
        device_P_ctx.resize(num_pipelines);
        
        for(int i = 0; i < num_pipelines; i++)
        {
            for(int j = 0; j < subgraph_generator->device_subNN_num; j++)
            {
                device_T_ctx[i].push_back(new nts::ctx::NtsContext());
            }
            for(int j = 0; j < subgraph_generator->device_num; j++)
            {
                device_P_ctx[i].push_back(new nts::ctx::NtsContext());
            }
        }

        for (int i_i = 0; i_i < iterations; i_i++) 
        {
            double epoch_time = 0.0;
            epoch_time -= get_time();

            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for(int i = 0; i < subgraph_generator->device_subNN_num; i++) {
                    for(int j = 0; j < graph->gnnctx->layer_size.size() - 1; j++) {
                        subgraph_generator->subNN_queq[i]->P[j]->zero_grad();   
                    }
                }
            }

            for(int i = 0; i < num_pipelines; i++) {
                for(int j = 0; j < subgraph_generator->device_subNN_num; j++) {
                    device_T_ctx[i][j]->train();
                }
                for(int j = 0; j < subgraph_generator->device_num; j++) {
                    device_P_ctx[i][j]->train();
                }
            }

            
            std::vector<std::thread> pipeline_threads;
            for(int pipeline_id = 0; pipeline_id < num_pipelines; pipeline_id++)
            {
                pipeline_threads.emplace_back([&](int pip_id){                    
                    decoupled_Forward(pip_id);
                }, pipeline_id);
            }
            for(int pipeline_id = 0; pipeline_id < num_pipelines; pipeline_id++) {
                pipeline_threads[pipeline_id].join();
            }

            epoch_time += get_time();
            per_epoch_time.push_back(epoch_time);

            acc(0, i_i, true);
            acc(1, i_i, true);
            acc(2, i_i, true);

        }

        double all_nn_time = 0.0;
        double all_T2P_time = 0.0;
        double all_g_time = 0.0;
        
        for(int i_i = 1; i_i < iterations; i_i++)
        {
            all_nn_time += NN_time[i_i];
            all_T2P_time += T2P_time[i_i];
            all_g_time += g_time[i_i];
            all_epoch_time += per_epoch_time[i_i];
        }
        std::cout << "--------------------finish algorithm !" << std::endl;
        std::cout << "all_nn_time: " << all_nn_time << std::endl;
        std::cout << "all_T2P_time: " << all_T2P_time << std::endl;
        std::cout << "all_g_time: " << all_g_time << std::endl;
        std::cout << "all_epoch_time: " << all_epoch_time << std::endl;
        std::cout << "max train acc: " << max_train_acc<< std::endl;
        std::cout << "max val acc: " << max_val_acc<< std::endl;
        std::cout << "max test acc: " << max_test_acc<< std::endl;

    }
    
    void run(){

        if(resourse_decoupled)
        {
            init_decoupled_version();
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