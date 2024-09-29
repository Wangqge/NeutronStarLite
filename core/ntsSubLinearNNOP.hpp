
#ifndef SUBLINEARNNOP_HPP
#define SUBLINEARNNOP_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "NtsScheduler.hpp"

namespace nts {
namespace op {

class SubLinearMemCostNNOP{
public:
  NtsVar *f_input;
  std::function<NtsVar(NtsVar &)>* forward_function;
  
  SubLinearMemCostNNOP(std::function<NtsVar(NtsVar &)> vertexforward){
      forward_function=(&vertexforward);
  }
  NtsVar forward(NtsVar &f_input_msg){// input i_msg  output o_msg
     NtsVar f_input_=f_input_msg.detach();
    f_input=&f_input_msg;
    return (*forward_function)(f_input_);
  }
  NtsVar backward(NtsVar &f_output_grad){// input i_msg  output o_msg
     //NtsVar f_input_=f_input.detach();
    NtsVar f_output=(*forward_function)(*f_input);
    f_output.backward(f_output_grad);
    return f_input->grad();
    
  }
};

} // namespace graphop
} // namespace nts

#endif
