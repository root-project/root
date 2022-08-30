#ifndef TMVA_SOFIE_RFUNCTION_SUM
#define TMVA_SOFIE_RFUNCTION_SUM


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

class RFunction_Sum: public RFunction_Aggregate{
    
    private:
        /*
        num_elements represents the number of collections used for aggregation in different aggregate relation
        For Edge_Node relation, it is number of edges
        For Edge_Globals relation, it is number of edges
        For Node_Globals relation, it is number of nodes
        */
        int num_elements; 
    public:
        RFunction_Sum(FunctionRelation relation,int NumElements):
        RFunction_Aggregate(relation), num_elements(NumElements){}

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
            for(int i=0; i<num_elements;++i){
                op_concat.reset(new ROperator_Concat<float>(fInputTensors[i],0,0,fFuncName+"InputConcatFeature"+std::to_string(i)));
                function_block->AddOperator(std::move(op_concat));
            }

            std::vector<std::string> Input_Stack;
            for(int i=0; i<num_elements; ++i){
                Input_Stack.emplace_back(fFuncName+"InputConcatFeature"+std::to_string(i));
            }

            op_concat.reset(new ROperator_Concat<float>(Input_Stack,0,1,fFuncName+"InputStack"));
            function_block->AddOperator(std::move(op_concat));

            std::unique_ptr<ROperator> op_reduce_sum;
            op_reduce_sum.reset(new ROperator_Reduce<float,EReduceOpMode::ReduceSum>(1,0,fFuncName+"InputStack","OutputTensor"));
            function_block->AddOperator(std::move(op_reduce_sum));

            function_block->AddOutputTensorNameList({"OutputTensor"});
        }

        void AddInputTensors(std::any inputShape){
            std::vector<std::vector<std::vector<std::size_t>>> fInputShape = std::any_cast<std::vector<std::vector<std::vector<std::size_t>>>>(inputShape); 
                for(long unsigned int i=0; i<fInputShape.size(); ++i){
                        for(long unsigned int j=0;j<fInputShape[0].size();++j){
                                function_block->AddInputTensorInfo(fInputTensors[i][j],ETensorType::FLOAT, fInputShape[i][j]);
                                function_block->AddInputTensorName(fInputTensors[i][j]);
                        }
                }
        }

};

} //SOFIE
} //Experimental
} //TMVA

#endif //TMVA_SOFIE_RFUNCTION_SUM