#ifndef TMVA_SOFIE_RFUNCTION_MLP
#define TMVA_SOFIE_RFUNCTION_MLP


#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator_Concat.hxx"
#include "TMVA/ROperator_Gemm.hxx"
#include "TMVA/ROperator_Relu.hxx"
#include "TMVA/RFunction.hxx"
#include "TMVA/RModel_GNN.hxx"

#include <any>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RFunction_MLP: public RFunction_Update{
    private:
        Int_t fNumLayers;          // Number of Layers in MLP
        bool fUseActivation;       // if True, ReLU is used as activation for every layer of the MLP

        std::vector<std::string> fKernelTensors;
        std::vector<std::string> fBiasTensors;

    public:
        RFunction_MLP(FunctionTarget target, Int_t numLayers, bool useActivation):
        RFunction_Update(target), fNumLayers(numLayers), fUseActivation(useActivation){}
        
        void Initialize(){
                        
            if(fTarget == FunctionTarget::EDGES){
                fInputTensors = {"edge","receiver","sender","global"};
            } else if(fTarget == FunctionTarget::NODES || fTarget == FunctionTarget::GLOBALS){
                fInputTensors = {"edge","node","global"}; 
            }

            std::unique_ptr<ROperator> op_concat;
            op_concat.reset(new ROperator_Concat<float>(fInputTensors,0,0,fFuncName+"InputConcat"));
            function_block->AddOperator(std::move(op_concat));
            
            std::unique_ptr<ROperator> op_gemm;
            std::string fGemmInput = fFuncName+"InputConcat";
            for(int i=0; i<fNumLayers; ++i){
                op_gemm.reset(new ROperator_Gemm<float>(1.0,1.0,0,0,fGemmInput,fKernelTensors[i],fBiasTensors[i],fFuncName+"Gemm"+std::to_string(i)));
                function_block->AddOperator(std::move(op_gemm));
                fGemmInput = fFuncName+"Gemm"+i;
                if(fUseActivation){
                    std::unique_ptr<ROperator> op_relu;
                    op_relu.reset(new ROperator_Relu<float>(fFuncName+"Gemm"+std::to_string(i), fFuncName+"Relu"+std::to_string(i)));
                    function_block->AddOperator(std::move(op_relu));
                    fGemmInput = fFuncName+"Relu"+i;
                }
            }
            function_block->AddBlasRoutines({"Gemm", "Gemv"});  // for Gemm operation

            // assuming all the linear layers has a kernel and a bias initialized tensors
            if(fUseActivation){
                function_block->AddOutputTensorNameList({fFuncName+"Relu"+std::to_string(fNumLayers-1)});
            } else{
                function_block->AddOutputTensorNameList({fFuncName+"Gemm"+std::to_string(fNumLayers-1)});
            }
        }

        void AddInitializedTensors(std::any initialized_tensors){
                std::vector<std::vector<std::string>> weight_tensors = std::any_cast<std::vector<std::vector<std::string>>>(initialized_tensors);
                fKernelTensors = weight_tensors[0];
                fBiasTensors   = weight_tensors[1];
        }
};

} // SOFIE
} // Experimental
} // TMVA

#endif //TMVA_SOFIE_RFUNCTION_MLP