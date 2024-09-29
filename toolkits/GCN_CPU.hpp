#include "core/neutronstar.hpp"

class GCN_CPU_impl {
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
  
  VertexSubset *active;
  
  Graph<Empty> *graph;
  
  
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  
  PartitionedGraph *partitioned_graph;
  
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  
  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  torch::nn::Dropout drpmodel;
  std::vector<torch::nn::BatchNorm1d> bn1d;
  
  double exec_time = 0;
  double all_sync_time = 0;
  double sync_time = 0;
  double all_graph_sync_time = 0;
  double graph_sync_time = 0;
  double all_compute_time = 0;
  double compute_time = 0;
  double all_copy_time = 0;
  double copy_time = 0;
  double graph_time = 0;
  double all_graph_time = 0;

  GCN_CPU_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;
    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = false;
    graph->rtminfo->lock_free = graph->config->lock_free;
  }
  void init_graph() {

    
    partitioned_graph=new PartitionedGraph(graph, active);
    partitioned_graph->GenerateAll([&](VertexId src, VertexId dst) {
      return nts::op::nts_norm_degree(graph,src, dst);
    },CPU_T,(graph->partitions)>1);
    graph->init_communicatior();
    
    ctx=new nts::ctx::NtsContext();
  }
  void init_nn() {

    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx, graph);
    
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                       graph->config->label_file,
                                       graph->config->mask_file);
      
      
      
    }

    
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);

    
    
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                beta2, epsilon, weight_decay));
      if(i < graph->gnnctx->layer_size.size() - 2)
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i])); 
    }

    
    
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
    }
    drpmodel = torch::nn::Dropout(
        torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(
        gnndatum->local_feature,
        {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        torch::DeviceType::CPU);

    
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }
    
    
    X[0] = F.set_requires_grad(true);
  }

  void Test(long s) { 
    NtsVar mask_train = MASK.eq(s);
    NtsVar all_train =
        X[graph->gnnctx->layer_size.size() - 1]
            .argmax(1)
            .to(torch::kLong)
            .eq(L_GT_C)
            .to(torch::kLong)
            .masked_select(mask_train.view({mask_train.size(0)}));
    NtsVar all = all_train.sum(0);
    long *p_correct = all.data_ptr<long>();
    long g_correct = 0;
    long p_train = all_train.size(0);
    long g_train = 0;
    MPI_Datatype dt = get_mpi_data_type<long>();
    MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    float acc_train = 0.0;
    if (g_train > 0)
      acc_train = float(g_correct) / g_train;
    if (graph->partition_id == 0) {
      if (s == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
      }
    }
  }
  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    
    if (layer == 0) {
      a =this->bn1d[layer](a);
      y = torch::relu(P[layer]->forward(a)).set_requires_grad(true);
    } else if (layer == 1) {
      y = P[layer]->forward(a);
      y = y.log_softmax(1);
    }
    
 
    return y;
  }
  void Loss() {
    
    
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1].log_softmax(-1);
    torch::Tensor mask_train = MASK.eq(0);
    loss = torch::nll_loss(
        a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
            .view({-1, a.size(1)}),
        L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
    ctx->appendNNOp(X[graph->gnnctx->layer_size.size() - 1], loss);
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      
        std::cout << "-------- W.grad sum : " << P[i]->W.grad().abs().sum().item<float>() << std::endl;
        std::cout << "-------- W.grad mean : " << P[i]->W.grad().abs().mean().item<float>() << std::endl;
        std::cout << "-------- W.grad max : " << P[i]->W.grad().abs().max().item<float>() << std::endl;
      
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }
  void Forward() {
    graph->rtminfo->forward = true;
    ctx->train();
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;


        std::cout << "-------- X sum : " << X[i].abs().sum().item<float>() << std::endl;
        std::cout << "-------- X mean : " << X[i].abs().mean().item<float>() << std::endl;
        std::cout << "-------- X max : " << X[i].abs().max().item<float>() << std::endl;

       NtsVar Y_i= ctx->runGraphOp<nts::op::ForwardCPUfuseOp>(partitioned_graph,active,X[i]);    

        std::cout << "-------- Y_i sum : " << Y_i.abs().sum().item<float>() << std::endl;
        std::cout << "-------- Y_i mean : " << Y_i.abs().mean().item<float>() << std::endl;
        std::cout << "-------- Y_i max : " << Y_i.abs().max().item<float>() << std::endl;

        X[i + 1]=ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
            if(i<(graph->gnnctx->layer_size.size() - 2)){
                n_i =this->bn1d[i](n_i);
                
                return torch::dropout(torch::relu(P[i]->forward(n_i)), drop_rate, ctx->is_train());
            }else{
                return  P[i]->forward(n_i);
            }
        },
        Y_i,
        X[i]);

        std::cout << "-------- W sum : " << P[i]->W.abs().sum().item<float>() << std::endl;
        std::cout << "-------- W mean : " << P[i]->W.abs().mean().item<float>() << std::endl;
        std::cout << "-------- W max : " << P[i]->W.abs().max().item<float>() << std::endl;

        
        std::cout << "-------- X+1 sum : " << X[i+1].abs().sum().item<float>() << std::endl;
        std::cout << "-------- X+1 mean : " << X[i+1].abs().mean().item<float>() << std::endl;
        std::cout << "-------- X+1 max : " << X[i+1].abs().max().item<float>() << std::endl;
    }
  }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n",
               iterations);
    }

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }
      
      Forward();
      Test(0);
      Test(1);
      Test(2);
      Loss();
      
      ctx->self_backward(false);
      Update();

      if (graph->partition_id == 0)
        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
                  << std::endl;
    }
    exec_time += get_time();





    





    delete active;
  }

};
