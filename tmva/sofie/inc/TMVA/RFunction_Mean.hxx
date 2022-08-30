#ifndef TMVA_SOFIE_RFUNCTION_MEAN
#define TMVA_SOFIE_RFUNCTION_MEAN


#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator_Concat.hxx"
#include "TMVA/ROperator_Reduce.hxx"
#include "TMVA/RFunction.hxx"
#include "TMVA/RModel_GNN.hxx"

#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RFunction_Mean: public RFunction_Aggregate{
    
    private:
        int num_elements;
    public:
        RFunction_Mean(FunctionRelation relation,int NumElements):
        RFunction_Aggregate(relation),num_elements(NumElements){}
        void Initialize(){
            std::vector<std::string> elementTensor;
            for(int i=0; i<num_elements; ++i){
                if(fRelation == FunctionRelation::EDGES_NODES){
                        elementTensor.emplace_back("Edge_"+std::to_string(i));
                        elementTensor.emplace_back("Receiver_"+std::to_string(i));
                        elementTensor.emplace_back("Sender_"+std::to_string(i));
                } else if(fRelation == FunctionRelation::EDGES_GLOBALS){
                        elementTensor.emplace_back("Edge_"+std::to_string(i));
                        elementTensor.emplace_back("Receiver_"+std::to_string(i));
                        elementTensor.emplace_back("Sender_"+std::to_string(i));
                } else if(fRelation == FunctionRelation::NODES_GLOBALS){
                        elementTensor.emplace_back("Node_"+std::to_string(i));
                } else{
                    throw std::runtime_error("Invalid relation for Aggregate function");
                }
                fInputTensors.emplace_back(elementTensor);
                elementTensor.clear();
            }
            
            std::unique_ptr<ROperator> op_concat;
            for(long unsigned int i=0; i<num_elements;++i){
                op_concat.reset(new ROperator_Concat<float>(fInputTensors[i],0,0,fFuncName+"InputConcatFeature"+std::to_string(i)));
                function_block->AddOperator(std::move(op_concat));
            }
            std::vector<std::string> Input_Stack;
            for(int i=0; i<num_elements; ++i){
                Input_Stack.emplace_back(fFuncName+"InputConcatFeature"+std::to_string(i));
            }
            op_concat.reset(new ROperator_Concat<float>(Input_Stack,0,1,fFuncName+"InputStack"));
            function_block->AddOperator(std::move(op_concat));

            std::unique_ptr<ROperator> op_reduce_mean;
            op_reduce_mean.reset(new ROperator_Reduce<float,EReduceOpMode::ReduceMean>(1,0,fFuncName+"InputStack","OutputTensor"));
            function_block->AddOperator(std::move(op_reduce_mean));

            function_block->AddOutputTensorNameList({"OutputTensor"});
        }
                
};

} //SOFIE
} //Experimental
} //TMVA

#endif //TMVA_SOFIE_RFUNCTION_MEAN