#include "core/neutronstar.hpp"

class GCN_GPUs_impl {
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
    
    
    VertexSubset *active;
    
    Graph<Empty> *graph;
    FullyRepGraph* full_rep_graph;

    MultiDeviceGenerator* subgraph_generator; 
    
    
    GNNDatum *gnndatum;
    
    
    

    
    std::vector<Parameter *> P;
    std::vector<NtsVar> X;
    nts::ctx::NtsContext* ctx;
    std::vector<nts::ctx::NtsContext *> device_ctx;
    
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


    
    GCN_GPUs_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false){
        graph = graph_;
        iterations = iterations_;

        LOG_INFO("config gpu num: %d", graph->config->gpu_num);
        num_devices = graph->config->gpu_num;
        if(num_devices == -1) {
            cudaGetDeviceCount(&num_devices);
        }
        assert(num_devices >= 0);

        LOG_INFO("config pipeline num: %d", graph->config->pipeline_num);
        num_pipelines = graph->config->pipeline_num;
        if(num_pipelines <= 0) {
            num_pipelines = 3;
        }
        assert(num_pipelines >= 1);

        active = graph->alloc_vertex_subset();
        active->fill();
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

        
        
        
        

        if(num_devices == 0){
            graph->rtminfo->with_cuda = false;
            init_CPU_version();
        }
        else {
            graph->rtminfo->with_cuda = true;
            if(num_devices > 1){
                init_multiGPU_version();
            } else {
                init_singleGPU_version();
            }
        }

        
    }

    void init_CPU_version(){  
        subgraph_generator = new MultiDeviceGenerator(full_rep_graph, gnndatum, num_devices);
        subgraph_generator->CPU_version();
    }

    void init_singleGPU_version(){

    }
  
    void init_multiGPU_version(){
        train_correct.resize(num_devices, 0);
        val_correct.resize(num_devices, 0);
        test_correct.resize(num_devices, 0);

        subgraph_generator = new MultiDeviceGenerator(full_rep_graph, gnndatum, num_devices);
        subgraph_generator->multi_GPU_version();

        std::vector<int> arr(num_devices);
        std::iota(arr.begin(), arr.end(), 0);
        NCCL_Communicator *nccl_comm = new NCCL_Communicator(num_devices, arr.data());

        for(int i = 0; i < num_devices; i++) {
            for(int j = 0; j < graph->gnnctx->layer_size.size() - 1; j++) {
                subgraph_generator->subgraph_queq[i]->P.push_back(new Parameter(graph->gnnctx->layer_size[j], graph->gnnctx->layer_size[j + 1], 
                                                                                alpha, beta1, beta2, epsilon, weight_decay));
                subgraph_generator->subgraph_queq[i]->P[j]->W.set_data(P[j]->W);
                subgraph_generator->subgraph_queq[i]->P[j]->set_decay(decay_rate, decay_epoch);

                subgraph_generator->subgraph_queq[i]->P[j]->set_multi_gpu_comm(nccl_comm);
            }
            subgraph_generator->subgraph_queq[i]->init_X_P(graph->config->K+1);
        }

        
        
        
        subgraph_generator->init_communicator(nccl_comm);
        subgraph_generator->load_data_to_corresponding_device();

    }

    NtsVar LOSS(NtsVar &predict, NtsVar &label, nts::ctx::NtsContext* ctx, NtsVar &mask)
    {
        
        
        
        
        
        
        

        torch::Tensor a = predict.log_softmax(-1);
        NtsVar mask_train = mask.eq(0);
        auto loss_train = torch::nll_loss(
            a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
                   .view({-1, a.size(1)}),
            label.masked_select(mask_train.view({mask_train.size(0)}))
        );
        if(ctx->training == true)
        {
            ctx->appendNNOp(predict, loss_train);
        }
        LOG_INFO("loss:%f", loss_train.item<float>());

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
            subgraph->P[i]->reset_gradient();
            subgraph->P[i]->learnG_with_decay_Adam();
            subgraph->P[i]->next();
        }
        for (int i = 0; i < subgraph->P.size(); i++) {
           subgraph->P[i]->zero_grad();
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
                    << per_epoch_time[epoch] << "(s)]" << std::endl;
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

    void CPU_Version()
    {
        LOG_INFO("GNNmini::[Single.CPU.GCNimpl] running [%d] Epoches\n",
               iterations);
        for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
            NtsVar d;
            X.push_back(d);
        }
        X[0] = subgraph_generator->subgraph_queq[0]->feature.set_requires_grad(true);
        
        for (int i_i = 0; i_i < iterations; i_i++) 
        {
            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for (int i = 0; i < P.size(); i++) {
                    P[i]->zero_grad();  
                }
            }

            
            graph->rtminfo->forward = true;
            for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
            graph->rtminfo->curr_layer = i;

            
            
            
            
            
            
            
            }
        }
        
    }

    void Multi_Forward(DeviceSubStructure *subgraph)
    {
        
        
        graph->rtminfo->forward = true;
        VertexId dev_id = subgraph->device_id;

        train_correct[dev_id] = 0;
        val_correct[dev_id] = 0;
        test_correct[dev_id] = 0;
        
        at::cuda::setCurrentCUDAStream(*(subgraph->ts));
        cudaSetUsingDevice(dev_id);
        for(int layer = 0; layer < (graph->gnnctx->layer_size.size()-1); layer++)
        {
            graph->rtminfo->curr_layer = layer;

            
            
            
            
            
            
            
            
            

            subgraph->X[layer+1] = device_ctx[dev_id]->runVertexForward([&](NtsVar n_i, NtsVar v_i){
                    return torch::dropout(torch::relu(subgraph->P[layer]->forward(n_i)), drop_rate, device_ctx[dev_id]->is_train());
            }, subgraph->X[layer], subgraph->X[layer]);
        }

        subgraph->X_P[0] = subgraph->X[graph->gnnctx->layer_size.size()-1];

        for(int layer = 0; layer < graph->config->K; layer++)
        {
            subgraph->X_P[layer+1] =  device_ctx[dev_id]->runGraphOp<nts::op::MultiGPUAllGNNCalcGraphSumOp>(subgraph, subgraph->X_P[layer]);
        }

        NtsVar loss = LOSS(subgraph->X_P[graph->config->K], subgraph->label_gpu, device_ctx[dev_id], 
                            subgraph->mask_gpu);

        
        

        
        train_correct[dev_id] = getCorrect(0, subgraph->X[graph->gnnctx->layer_size.size()-1], 
                                            subgraph->label_gpu, subgraph->mask_gpu);
        val_correct[dev_id] = getCorrect(1, subgraph->X[graph->gnnctx->layer_size.size()-1], 
                                            subgraph->label_gpu, subgraph->mask_gpu);
        test_correct[dev_id] = getCorrect(2, subgraph->X[graph->gnnctx->layer_size.size()-1], 
                                            subgraph->label_gpu, subgraph->mask_gpu);

        device_ctx[dev_id]->self_backward(false);

        Update(subgraph);

    }
    
    void Multi_GPU_Version()
    {
        LOG_INFO("GNNmini::[MultiGPU[%d].CPU.GCNimpl] running [%d] Epoches\n", num_devices, iterations);
        device_ctx.resize(num_devices);
        for(int i = 0; i < num_devices; i++) {
            device_ctx[i] = new nts::ctx::NtsContext();
        }

        for (int i_i = 0; i_i < iterations; i_i++) 
        {
            double epoch_time = 0.0;
            epoch_time -= get_time();

            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for(int i = 0; i < num_devices; i++) {
                    for(int j = 0; j < graph->gnnctx->layer_size.size() - 1; j++) {
                        subgraph_generator->subgraph_queq[i]->P[j]->zero_grad();   
                    }
                }
            }

            for(int device_id = 0; device_id < num_devices; device_id++) {
                device_ctx[device_id]->train();
            }

            
            std::vector<std::thread> device_threads;
            for(int device_id = 0; device_id < num_devices; device_id++)
            {
                device_threads.emplace_back([&](int dev_id){                    
                    assert(dev_id == subgraph_generator->subgraph_queq[dev_id]->device_id);
                    Multi_Forward(subgraph_generator->subgraph_queq[dev_id]);
                }, device_id);
            }
            for(int device_id = 0; device_id < num_devices; device_id++) {
                device_threads[device_id].join();
            }

            epoch_time += get_time();
            per_epoch_time.push_back(epoch_time);

            acc(0, i_i, true);
            acc(1, i_i, true);
            acc(2, i_i, true);

        }

        for(auto&& t : per_epoch_time)
        {
            all_epoch_time += t;
        }
        std::cout << "--------------------finish algorithm !" << std::endl;
        std::cout << "all_epoch_time: " << all_epoch_time << std::endl;
        std::cout << "max train acc: " << max_train_acc<< std::endl;
        std::cout << "max val acc: " << max_val_acc<< std::endl;
        std::cout << "max test acc: " << max_test_acc<< std::endl;
        

    }

    void run(){
        if(num_devices == 0)
        {
            CPU_Version();
        } else if (num_devices == 1)
        {
            
        } else
        {
            Multi_GPU_Version();
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