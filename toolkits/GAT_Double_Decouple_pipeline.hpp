#include "core/neutronstar.hpp"

class GAT_Double_Decouple_pipeline_impl {
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

    int num_devices;//num_device = 0是CPU执行
    int num_pipelines;
    int K;
    ValueType appnp_alpha;
    
    // graph
    // VertexSubset *active;
    // graph with no edge data
    Graph<Empty> *graph;
    FullyRepGraph* full_rep_graph;

    HoriDecouplePartition *pipeline_partitioner;
    
    //NN
    GNNDatum *gnndatum;
    // NtsVar label_cpu;//这三个东西在子图那个类里面了
    // NtsVar label_gpu;
    // NtsVar MASK;

    // Variables
    std::vector<Parameter *> P;
    std::vector<Parameter *> PE;
    NtsVar PE0;
    std::vector<std::vector<Parameter *>> device_NN_P;
    std::vector<NtsVar> X;
    std::vector<std::vector<NtsVar>> X_device;
    nts::ctx::NtsContext* ctx;
    std::vector<nts::ctx::NtsContext *> chunk_ctx;
    std::vector<std::vector<VertexId>> layer_per_gpu;

    std::vector<std::vector<nts::ctx::NtsContext *>> device_T_ctx;//for decoupled  [pipeline id][device id]
    std::vector<std::vector<nts::ctx::NtsContext *>> device_P_ctx;//for decoupled


    
    NtsVar F;
    // NtsVar loss;
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

    //time
    std::vector<std::vector<double>> NN_time;
    std::vector<std::vector<double>> g_time;
    std::vector<std::vector<double>> T2P_time;
    std::vector<std::vector<double>> loss_time;
    std::vector<std::vector<double>> gB_time;
    std::vector<std::vector<double>> P2T_time;
    std::vector<std::vector<double>> nnB_time;
    std::vector<std::vector<double>> pip_time;


    std::mutex T_forward_mutex;
    std::mutex P_mutex;
    std::mutex T_backward_mutex;
    std::queue<int> T_forward_queue;
    std::queue<int> P_queue;
    std::queue<int> T_backward_queue;
    
    double no_P_time = 0.0;
    
    GAT_Double_Decouple_pipeline_impl(Graph<Empty> *graph_, int iterations_,
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

        // active = graph->alloc_vertex_subset();
        // active->fill();
        graph->init_gnnctx(graph->config->layer_string);
        //APPNP不采样
        // graph->init_gnnctx_fanout(graph->config->fanout_string);
        // rtminfo initialize
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
        if(num_devices > 1){//多GPU要创建多个ctx，还没写

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
        // gnndatum->registLabel(label_cpu);
        // gnndatum->registMask(MASK);

        for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
            P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                        graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                        beta2, epsilon, weight_decay));
        }

        PE.push_back(new Parameter(graph->gnnctx->layer_size[graph->gnnctx->layer_size.size() - 1] * 2, 1, alpha, beta1,
                                        beta2, epsilon, weight_decay));

        for (int i = 0; i < P.size(); i++) {
            P[i]->init_parameter();
            P[i]->set_decay(decay_rate, decay_epoch);
        }

        PE[0]->init_parameter();
        PE[0]->set_decay(decay_rate, decay_epoch);


        // F = graph->Nts->NewLeafTensor( //
        //     gnndatum->local_feature,
        //     {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        //     torch::DeviceType::CPU);


        // debug_init_info();
    }

    void init_decoupled_version(){
        //根据pipeline num做图划分，删除子图之间的依赖
        pipeline_partitioner = new HoriDecouplePartition(full_rep_graph, gnndatum, num_pipelines, num_devices);
        pipeline_partitioner->partition_to_chunk();
        pipeline_partitioner->compute_P_T_GPUs(graph->gnnctx->layer_size.size()-1, K);
        pipeline_partitioner->alloc_GPU_handle_chunk();
        
        // layer_per_gpu.resize(num_pipelines);
        // for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
        // {
        //     layer_per_gpu[pip_id].resize(num_devices, 0);
        // }

        //就先写一个GPU做NN的了
        for(int i = 0; i < pipeline_partitioner->T_num; i++)
        {
            for(int j = 0; j < P.size(); j++)
            { 
                torch::Device GPU(torch::kCUDA, i);
                P[j]->to(GPU);
                P[j]->Adam_to_GPU();
            }

            torch::Device GPU(torch::kCUDA, i);
            PE[0]->to(GPU);
            PE[0]->Adam_to_GPU();
        }
        for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
        {
            // init all communicator
            std::vector<int> arr_(num_devices);
            std::iota(arr_.begin(), arr_.end(), 0);
            NCCL_Communicator *nccl_comm_all = new NCCL_Communicator(num_devices, arr_.data());

            //每一个chunk应该拥有一个通信器、四个cuda流
            pipeline_partitioner->chunks[pip_id]->init_cuda_and_comm(nccl_comm_all);
            if(graph->config->SMALLGRAPH == 1)
            {
                //load feature to gpu0,load graph to retain gpu
                pipeline_partitioner->chunks[pip_id]->load_feat_to_gpu0();
                pipeline_partitioner->chunks[pip_id]->load_graph_to_P_gpu_GAT(K);
            }
        }
        
        // for(int pip_id = 0; pip_id < num_pipelines; pip_id++)
        // {
        //     for(int k = 0; k < num_devices; k++)
        //     {
        //         std::cout << "chunk: " << pip_id << " gpu: " << k << " layer: " << layer_per_gpu[pip_id][k] << std::endl; 
        //     }
        // }
        // assert(false);
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
        // LOG_INFO("loss:%f", loss_train.item<float>());

        train_correct[pip_id] = getCorrect(0, predict, label, mask);
        val_correct[pip_id] = getCorrect(1, predict, label, mask);
        test_correct[pip_id] = getCorrect(2, predict, label, mask);

        return loss_train;
    }

    long getCorrect(long s, NtsVar &predict, NtsVar &label, NtsVar &mask) { // 0 train, //1 eval //2 test
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
            // std::cout << "P[" << i << "].grad : " << P[i]->W.grad().abs().sum() << std::endl;
            if(pip_id==0){
               P[i]->reset_gradient(P[i]->W.grad());
            }
            else{
                P[i]->store_gradient(P[i]->W.grad());
            }
            // std::cout << "P[" << i << "].W_gradient : " << P[i]->W_gradient.abs().sum() << std::endl;
        }
        // for (int i = 0; i < P.size(); i++) {
        //        P[i]->zero_grad();
        // }
    }

    void acc(int type, int epoch, bool log = false)//这个准确率是所有流水线中的正确的
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
                        << ") "<<",pip time:"<< pip_time[pip_id][epoch]<<std::endl;
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
                test_acc.push_back(acc);
                LOG_INFO("test Acc: %f %d %d", acc, correct_num, label_num);
            } else {
                test_acc.push_back(acc);
            }
        }


    }
    
    NtsVar vertexForward(NtsVar &x) {
        NtsVar y;
        int layer = graph->rtminfo->curr_layer;
        // if (layer == 0) {
            y = torch::relu(P[layer]->forward(x)).set_requires_grad(true);
        // } else if (layer == 1) {
        //     y = P[layer]->forward(x);
        //     y = y.log_softmax(1); // CUDA
        // }

        //cp->op_push(a, y, nts::autodiff::NNOP);
        return y;
    }
    
    void decoupled_Forward(int pip_id)
    {
        graph->rtminfo->forward = true;

        GraphChunk *graph_chunk = pipeline_partitioner->chunks[pip_id];

        double NN_t = 0.0;
        NN_t -= get_time();

        int layer = 0;

        //NN Forward, NN都使用第一个子图的参数，确保只有一个model
        for(int layer = 0; layer < (graph->gnnctx->layer_size.size()-1); layer++)
        {
            graph->rtminfo->curr_layer = layer;
            graph_chunk->X_T[layer + 1] = vertexForward(graph_chunk->X_T[layer]);
        }
        // graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        NN_t += get_time();
        NN_time[pip_id].push_back(NN_t);


        double T2P_t = 0.0;
        T2P_t -= get_time();

        //T device embedding transfar to P device
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
        P_t += get_time();
        g_time[pip_id].push_back(P_t);

        //loss
        //getcorrect //这个时间可以减掉
        double loss_t = 0.0;
        loss_t -= get_time();
        NtsVar loss = LOSS(graph_chunk->X_P[K], graph_chunk->label_gpu, graph_chunk->mask_gpu, pip_id);
        loss.backward(torch::ones_like(loss), false);

        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        loss_t += get_time();
        loss_time[pip_id].push_back(loss_t);

        //P backward
        double PB_t = 0.0;
        PB_t -= get_time();
        graph_chunk->X_PB[0] = graph_chunk->X_P[K].grad();
        for(int layer = 0; layer < K; layer++)
        {
            graph_chunk->SPMM_csr(layer);
        }
        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        PB_t += get_time();
        gB_time[pip_id].push_back(PB_t);

        // graph_chunk->debug_X();

        //P device grad transfar to T device
        double P2T_t = 0.0;
        P2T_t -= get_time();
        
        graph_chunk->send_grad(0);

        P2T_t += get_time();
        P2T_time[pip_id].push_back(P2T_t);
        
        //NN backward and update
        double nnB_t = 0.0;
        nnB_t -= get_time();
 
        // graph_chunk->debug_X();

        graph_chunk->X_T[graph_chunk->X_T.size() - 1].backward(graph_chunk->X_TB, true);
        add_gradient(pip_id);
        nnB_t += get_time();
        nnB_time[pip_id].push_back(nnB_t);        
    }
    
    void decoupled_T(int pip_id)
    {
        GraphChunk *graph_chunk = pipeline_partitioner->chunks[pip_id];

        double NN_t = 0.0;
        NN_t -= get_time();

        int layer = 0;

        //NN Forward, NN都使用第一个子图的参数，确保只有一个model
        for(int layer = 0; layer < (graph->gnnctx->layer_size.size()-1); layer++)
        {
            graph->rtminfo->curr_layer = layer;
            graph_chunk->X_T[layer + 1] = vertexForward(graph_chunk->X_T[layer]);
            if(layer == 0)
            {
                auto device =graph_chunk->X_T[layer].device();
                if(pip_id > 3)
                {
                    for (int m = 0; m < 2; m++)
                    {
                    
                        NtsVar ls = torch::mm(torch::randn({graph_chunk->X_T[layer].size(0), graph_chunk->X_T[layer + 1].size(1)}).to(device),
                                                torch::randn({graph_chunk->X_T[layer + 1].size(1), graph_chunk->X_T[layer + 1].size(1)}).to(device));
                    }
                }
                if(pip_id > 5)
                {
                    for (int m = 0; m < 2; m++)
                    {
                        NtsVar ls = torch::mm( torch::randn({graph_chunk->X_T[layer].size(0), graph_chunk->X_T[layer + 1].size(1)}).to(device),
                                                torch::randn({graph_chunk->X_T[layer + 1].size(1), graph_chunk->X_T[layer + 1].size(1)}).to(device));
                    }
                }
            }
        }
        // graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        NN_t += get_time();
        NN_time[pip_id].push_back(NN_t);

        //复制
        NtsVar X_trans = graph->Nts->NewKeyTensor(graph_chunk->X_T[graph_chunk->X_T.size() - 1].packed_accessor<ValueType, 2>().data(),
                                                    {graph_chunk->X_T[graph_chunk->X_T.size() - 1].size(0), 
                                                    graph_chunk->X_T[graph_chunk->X_T.size() - 1].size(1)});
        // std::cout << "X_T size : " << graph_chunk->X_T[graph_chunk->X_T.size() - 1].sizes() << "X_T sum: " << graph_chunk->X_T[graph_chunk->X_T.size() - 1].abs().sum() << std::endl;


        double T2P_t = 0.0;
        T2P_t -= get_time();

        //T device embedding transfar to P device
        graph_chunk->send_embedding();

        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        T2P_t += get_time();
        T2P_time[pip_id].push_back(T2P_t);




        // Edge NN

        // std::cout << "X_trans size : " << X_trans.sizes() << "X_trans sum: " << X_trans.abs().sum() << std::endl;


        NtsVar m;
        {
            torch::NoGradGuard no_grad;
            // LOG_INFO("1");
            auto device = X_trans.device();

            // std::cout << "X_trans size : " << X_trans.sizes() << "X_trans sum: " << X_trans.abs().sum() << std::endl;
            
            NtsVar edge_src = graph_chunk->ScatterSrc(torch::mm(X_trans, torch::randn({X_trans.size(1), 1}).to(device)));
            // std::cout << "edge_src size : " << edge_src.sizes() << "edge_src sum: " << edge_src.abs().sum() << std::endl;
            // LOG_INFO("2");
            NtsVar edge_dst = graph_chunk->ScatterDst(torch::mm(X_trans, torch::randn({X_trans.size(1), 1}).to(device)));
            // std::cout << "edge_dst size : " << edge_dst.sizes() << "edge_dst sum: " << edge_dst.abs().sum() << std::endl;
            // LOG_INFO("3");
            // NtsVar e_msg=torch::cat({edge_src,edge_dst},1);
            NtsVar e_msg= edge_src + edge_dst;
            // std::cout << "e_msg size : " << e_msg.sizes() << "e_msg sum: " << e_msg.abs().sum() << std::endl;
            // LOG_INFO("4");
            // add by lusz
            // auto device = e_msg.device();
            // NtsVar random_tensor = torch::randn({e_msg.sizes()[0], e_msg.sizes()[1]}).to(device);
            m = torch::leaky_relu(e_msg, 0.2);
        }
        // NtsVar m = torch::leaky_relu(PE[0]->forward(e_msg),0.2);
        // LOG_INFO("5");
        // std::cout << "m size : " << m.sizes() << "m sum: " << m.abs().sum() << std::endl;
        
        NtsVar a = graph_chunk->edge_softmax(m);
        // std::cout << "a size : " << a.sizes() << "a sum: " << a.abs().sum() << std::endl;

        // assert(false);
    }
    
    void decoupled_P(int pip_id)
    {
        GraphChunk *graph_chunk = pipeline_partitioner->chunks[pip_id];
        double P_t = 0.0;
        P_t -= get_time();
        for(int layer = 0; layer < K; layer++)
        {
            // LOG_INFO("start outer:%d",layer);
            graph_chunk->SPMM_csc(layer);
            if(layer == 0)
            {
                if(pip_id > 3)
                {
                    for (int m = 0; m < 2; m++)
                    {
                        graph_chunk->SPMM_csc(layer);
                    }
                }
                if(pip_id > 5)
                {
                    for (int m = 0; m < 2; m++)
                    {
                        graph_chunk->SPMM_csc(layer);
                    }
                }
            }
            // LOG_INFO("end outer:%d",layer);
        }
        
        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        P_t += get_time();
        g_time[pip_id].push_back(P_t);

        //loss
        //getcorrect //这个时间可以减掉
        double loss_t = 0.0;
        loss_t -= get_time();
        NtsVar loss = LOSS(graph_chunk->X_P[K], graph_chunk->label_gpu, graph_chunk->mask_gpu, pip_id);
        loss.backward(torch::ones_like(loss), false);

        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        loss_t += get_time();
        loss_time[pip_id].push_back(loss_t);

        //P backward
        double PB_t = 0.0;
        PB_t -= get_time();
        graph_chunk->X_PB[0] = graph_chunk->X_P[K].grad();
        for(int layer = 0; layer < K; layer++)
        {
            graph_chunk->SPMM_csr(layer);
        }
        graph_chunk->cs->CUDA_DEVICE_SYNCHRONIZE();
        PB_t += get_time();
        gB_time[pip_id].push_back(PB_t);

        //P device grad transfar to T device
        double P2T_t = 0.0;
        P2T_t -= get_time();
        
        graph_chunk->send_grad(0);

        P2T_t += get_time();
        P2T_time[pip_id].push_back(P2T_t);

    }
    
    void T_backward(int pip_id)
    {
        GraphChunk *graph_chunk = pipeline_partitioner->chunks[pip_id];
        //NN backward and update
        double nnB_t = 0.0;
        nnB_t -= get_time();
 
        // graph_chunk->debug_X();
        graph_chunk->X_T[graph_chunk->X_T.size() - 1].backward(graph_chunk->X_TB, false);
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
        pip_time.resize(num_pipelines);
        LOG_INFO("GNNmini::[decoupled.GPU.APPNPimpl] running [%d] Epoches\n", iterations);
        
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

            //forward
            std::function pipeline_T = [&]
                {
                    for(int i = 0; i < num_pipelines; i++)
                    {
                        // TIME
                        // add by lusz
                        double per_pip_time=0.0;
                        per_pip_time-=get_time();
                        if(graph->config->SMALLGRAPH==0){
                            pipeline_partitioner->chunks[i]->load_feat_to_gpu0();
                            LOG_INFO("pipid:%d, gpuid:%d",i,pipeline_partitioner->chunks[i]->gpu_id);
                            // pipeline_partitioner->chunks[pip_id]->gpu_id=pip_id%(graph->config->gpu_num-1)+1;
                            pipeline_partitioner->chunks[i]->load_graph_to_P_gpu_GAT(K);
                        }
                        decoupled_T(i);

                        T_forward_mutex.lock();
                        T_forward_queue.push(i);
                        T_forward_mutex.unlock();


                        //P
                        while(1)
                        {
                            T_forward_mutex.lock();
                            bool condition = !T_forward_queue.empty();
                            T_forward_mutex.unlock();
                            if(condition)
                            {
                                break;
                            }    
                            __asm volatile("pause" ::: "memory");
                        }
                        T_forward_mutex.lock();
                        int pipeline_id = T_forward_queue.front();
                        T_forward_queue.pop();
                        T_forward_mutex.unlock();

                        decoupled_P(pipeline_id);

                        P_mutex.lock();
                        P_queue.push(pipeline_id);
                        P_mutex.unlock();
                        //
                        if(graph->config->SMALLGRAPH==0){
                            pipeline_partitioner->chunks[i]->free_graph_from_P_gpu();
                        }


                        // TB
                        while(1)
                        {
                            P_mutex.lock();
                            bool condition = !P_queue.empty();
                            P_mutex.unlock();
                            if(condition)
                            {
                                break;
                            }
                            __asm volatile("pause" ::: "memory");
                        }
                        P_mutex.lock();
                        pipeline_id = P_queue.front();
                        P_queue.pop();
                        P_mutex.unlock();

                        T_backward(pipeline_id);
                        if(graph->config->SMALLGRAPH==0){
                            pipeline_partitioner->chunks[i]->free_feat_from_gpu0();
                        }
                        per_pip_time+=get_time();
                        pip_time[i].push_back(per_pip_time);
                        std::cout << "pip_time" <<pipeline_id<<":"<<pip_time[pipeline_id][i_i]<<std::endl;
                    }

                };

            std::function pipeline_P = [&]
                {
                    for(int step = 0; step < num_pipelines; step++)
                    {
                        while(1)
                        {
                            T_forward_mutex.lock();
                            bool condition = !T_forward_queue.empty();
                            T_forward_mutex.unlock();
                            if(condition)
                            {
                                break;
                            }    
                            __asm volatile("pause" ::: "memory");
                        }
                        T_forward_mutex.lock();
                        int pipeline_id = T_forward_queue.front();
                        T_forward_queue.pop();
                        T_forward_mutex.unlock();

                        decoupled_P(pipeline_id);

                        P_mutex.lock();
                        P_queue.push(pipeline_id);
                        P_mutex.unlock();
                        //
                        if(graph->config->SMALLGRAPH==0){
                            pipeline_partitioner->chunks[step]->free_graph_from_P_gpu();
                        }
                    }
                };

            std::function pipeline_T_B = [&]
                {
                    for(int step = 0; step < num_pipelines; step++)
                    {
                        while(1)
                        {
                            P_mutex.lock();
                            bool condition = !P_queue.empty();
                            P_mutex.unlock();
                            if(condition)
                            {
                                break;
                            }
                            __asm volatile("pause" ::: "memory");
                        }
                        P_mutex.lock();
                        int pipeline_id = P_queue.front();
                        P_queue.pop();
                        P_mutex.unlock();

                        T_backward(pipeline_id);
                        if(graph->config->SMALLGRAPH==0){
                            pipeline_partitioner->chunks[step]->free_feat_from_gpu0();
                        }
                    }
                };
            
            // std::thread T_forward_thread(pipeline_T);
            // std::thread P_thread(pipeline_P);
            // std::thread T_backward_thread(pipeline_T_B);

            // T_forward_thread.join();
            // P_thread.join();
            // T_backward_thread.join();

            pipeline_T();
            // pipeline_P();
            // pipeline_T_B();
            epoch_time += get_time();
            per_epoch_time.push_back(epoch_time);

            Update_P();//拿第一个子图的NN参数来做

            acc(0, i_i, true);
            acc(1, i_i, true);
            acc(2, i_i, true);

            // assert(false);

        }

        double all_nn_time = 0.0;
        double all_T2P_time = 0.0;
        double all_g_time = 0.0;

        CHECK_CUDA_RESULT(cudaDeviceSynchronize());
        all_epoch_time = 0;
        for(int i_i = 1; i_i < iterations; i_i++)
        {
            // all_nn_time += NN_time[i_i];
            // all_T2P_time += T2P_time[i_i];
            // all_g_time += g_time[i_i];
            all_epoch_time += per_epoch_time[i_i];
        }
        std::cout << "--------------------finish algorithm !" << std::endl;
        // std::cout << "all_nn_time: " << all_nn_time << std::endl;
        // std::cout << "all_T2P_time: " << all_T2P_time << std::endl;
        // std::cout << "all_g_time: " << all_g_time << std::endl;
        std::cout << "all_epoch_time: " << all_epoch_time/iterations << std::endl;
        std::cout << "max train acc: " << max_train_acc<< std::endl;
        std::cout << "max val acc: " << max_val_acc<< std::endl;
        std::cout << "max test acc: " << max_test_acc<< std::endl;

        // // 画图
        // std::string prefix = "output_pipeline_";
        // std::string DatasetName="reddit";
        // std::string filepath="./outputs/";
        // std::string extension = ".txt";
        
        // std::string filename = filepath+DatasetName+"/"+prefix+DatasetName+std::to_string(num_pipelines)+extension;
        // if (FILE *file = fopen(filename.c_str(), "r")) {
        //     fclose(file);
        //     // 如果存在，则删除文件
        //     if (remove(filename.c_str()) != 0) {
        //         std::cerr << "无法删除文件：" << filename << std::endl;
        //     } else {
        //     std::cout << "已删除现有文件：" << filename << std::endl;
        //     }
        // }
        // std::ofstream outfile(filename);
        // if (!outfile.is_open()) {
        //     std::cerr << "无法打开文件 " << filename << " !" << std::endl;
        // }

        // // 将 train_correct 中的元素输出到文件中
        // for (ValueType element : test_acc) {
        //     outfile << element << " ";
        //     // std::cout<<element<<std::endl;
        // }
        // outfile.close();
        // std::cout << "数据已写入文件 " << filename << std::endl;

    }
    
    void run(){
        if(resourse_decoupled)
        {
            init_decoupled_version();
            run_decoupled_version();
        }

        // else{
        //     if(num_devices == 0){
        //         graph->rtminfo->with_cuda = false;
        //         init_CPU_version();
        //         CPU_Version();
        //     }
        //     else {
        //         graph->rtminfo->with_cuda = true;
        //         if(num_devices > 1){
        //             init_multiGPU_version();
        //             run_Multi_GPU_Version();
        //         } else {
        //             init_singleGPU_version();
        //         }
        //     }
        // }
        

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