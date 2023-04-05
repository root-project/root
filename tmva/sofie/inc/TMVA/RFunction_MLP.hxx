#ifndef TMVA_SOFIE_RFUNCTION_MLP
#define TMVA_SOFIE_RFUNCTION_MLP


#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator_Concat.hxx"
#include "TMVA/ROperator_Gemm.hxx"
#include "TMVA/ROperator_LayerNormalization.hxx"
#include "TMVA/ROperator_Relu.hxx"
#include "TMVA/RFunction.hxx"
#include "TMVA/RModel_GNN.hxx"

#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>
#include <cassert>
#include <memory>
#include <vector>

#include <iostream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum class Activation {
   RELU = 0x0,
   Invalid = 0x1,
};

class RFunction_MLP: public RFunction_Update{
    private:
        Int_t fNumLayers;           // Number of Layers in MLP
        Activation fActivationFunction;
        bool  fActivateFinal;       // if True, fActivationFunction is applied as the activation for the last layer
        std::vector<std::string> fKernelTensors;
        std::vector<std::string> fBiasTensors;

    public:
        virtual ~RFunction_MLP(){}
        RFunction_MLP(FunctionTarget target, Int_t numLayers, Activation activation_function=Activation::RELU, bool activate_final=false, GraphType gType=GraphType::GNN):
        RFunction_Update(target, gType), fNumLayers(numLayers), fActivationFunction(activation_function), fActivateFinal(activate_final){
            if(fActivationFunction == Activation::Invalid){
                throw std::runtime_error("TMVA SOFIE GNN doesn't currently supports the provided activation function for " + fFuncName + " update.");
            }

            // assuming all the linear layers has a kernel and a bias initialized tensors
            if(fActivateFinal){
                function_block->AddOutputTensorNameList({fFuncName+"Relu"+std::to_string(fNumLayers)});
            } else{
                function_block->AddOutputTensorNameList({fFuncName+"Gemm"+std::to_string(fNumLayers)});
            }
        }

        void Initialize(){
            
            std::string fGemmInput;
            if(fGraphType == GraphType::GNN){            
                std::unique_ptr<ROperator> op_concat;
                op_concat.reset(new ROperator_Concat<float>(fInputTensors,1,0,fFuncName+"InputConcat"));
                function_block->AddOperator(std::move(op_concat));
                fGemmInput = fFuncName+"InputConcat";

            } else if(fGraphType == GraphType::GraphIndependent){
                fGemmInput = fInputTensors[0];
            }

            std::unique_ptr<ROperator> op_gemm;
            for(int i=0; i<fNumLayers-1; ++i){
                op_gemm.reset(new ROperator_Gemm<float>(1.0,1.0,0,0,fGemmInput,UTILITY::Clean_name(fKernelTensors[i]),UTILITY::Clean_name(fBiasTensors[i]),fFuncName+"Gemm"+std::to_string(i)));
                function_block->AddOperator(std::move(op_gemm));
                fGemmInput = fFuncName+"Gemm"+i;
                if (fActivationFunction == Activation::RELU){
                    std::unique_ptr<ROperator> op_relu;
                    op_relu.reset(new ROperator_Relu<float>(fFuncName+"Gemm"+std::to_string(i), fFuncName+"Relu"+std::to_string(i)));
                    function_block->AddOperator(std::move(op_relu));
                    fGemmInput = fFuncName+"Relu"+i;

                }       
            }

            op_gemm.reset(new ROperator_Gemm<float>(1.0,1.0,0,0,fGemmInput,UTILITY::Clean_name(fKernelTensors.back()),UTILITY::Clean_name(fBiasTensors.back()),fFuncName+"Gemm"+std::to_string(fNumLayers)));
            function_block->AddOperator(std::move(op_gemm));
            if(fActivateFinal){
                if (fActivationFunction == Activation::RELU){
                    std::unique_ptr<ROperator> op_relu;
                    op_relu.reset(new ROperator_Relu<float>(fFuncName+"Gemm"+std::to_string(fNumLayers), fFuncName+"Relu"+std::to_string(fNumLayers)));
                    function_block->AddOperator(std::move(op_relu));
                } 
            }


            if(fAddlOp.size()){
                for(auto &i:fAddlOp){
                    std::unique_ptr<ROperator> tmp(i);
                    function_block->AddOperator(std::move(tmp));                
                }
            }
        }

        void AddLayerNormalization(int axis, float epsilon, size_t stashType, const std::string &nameX,
                                    const std::string &nameScale, const std::string &nameB, const std::string &nameY){
            auto op_layerNorm = new ROperator_LayerNormalization<float>(axis, epsilon, stashType, nameX,
                                                                        nameScale, nameB, nameY, "", "");
            fAddlOp.push_back((op_layerNorm));
        }
        

        void AddInitializedTensors(const std::vector<std::vector<std::string>>& initialized_tensors){
                fKernelTensors = initialized_tensors[0];
                fBiasTensors   = initialized_tensors[1];
        }
};

} // SOFIE
} // Experimental
} // TMVA

#endif //TMVA_SOFIE_RFUNCTION_MLP
