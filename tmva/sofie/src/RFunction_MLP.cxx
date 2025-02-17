#include "TMVA/RFunction_MLP.hxx"


#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator_Concat.hxx"
#include "TMVA/ROperator_Gemm.hxx"
#include "TMVA/ROperator_LayerNormalization.hxx"
#include "TMVA/ROperator_Relu.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

RFunction_MLP::RFunction_MLP(FunctionTarget target, Int_t numLayers, Activation activation_function, bool activate_final, GraphType gType):
    RFunction_Update(target, gType), fNumLayers(numLayers), fActivationFunction(activation_function), fActivateFinal(activate_final)
{
   // assuming all the linear layers has a kernel and a bias initialized tensors
   if (fActivateFinal) {
      if (fActivationFunction == Activation::Invalid) {
         throw std::runtime_error("TMVA SOFIE GNN doesn't currently supports the provided activation function for " +
                                  fFuncName + " update.");
      }
      function_block->AddOutputTensorNameList({fFuncName + "Relu" + std::to_string(fNumLayers)});
   } else {
      function_block->AddOutputTensorNameList({fFuncName + "Gemm" + std::to_string(fNumLayers)});
   }
}

void RFunction_MLP::Initialize() {

    std::string fGemmInput;
    if(fGraphType == GraphType::GNN) {
        std::unique_ptr<ROperator> op_concat;
        op_concat.reset(new ROperator_Concat(fInputTensors,1,0,fFuncName+"InputConcat"));
        function_block->AddOperator(std::move(op_concat));
        fGemmInput = fFuncName+"InputConcat";

    } else if(fGraphType == GraphType::GraphIndependent) {
        fGemmInput = fInputTensors[0];
    }

    std::unique_ptr<ROperator> op_gemm;
    for(int i=0; i<fNumLayers-1; ++i) {
        double beta = (fBiasTensors[i].empty()) ? 0. : 1.;
        op_gemm.reset(new ROperator_Gemm<float>(1.0,beta,0,0,fGemmInput,UTILITY::Clean_name(fKernelTensors[i]),UTILITY::Clean_name(fBiasTensors[i]),fFuncName+"Gemm"+std::to_string(i)));
        function_block->AddOperator(std::move(op_gemm));
        fGemmInput = fFuncName+"Gemm"+i;
        if (fActivationFunction == Activation::RELU) {
            std::unique_ptr<ROperator> op_relu;
            op_relu.reset(new ROperator_Relu<float>(fFuncName+"Gemm"+std::to_string(i), fFuncName+"Relu"+std::to_string(i)));
            function_block->AddOperator(std::move(op_relu));
            fGemmInput = fFuncName+"Relu"+i;

        }
    }
    double beta = (fBiasTensors.back().empty()) ? 0. : 1.;
    op_gemm.reset(new ROperator_Gemm<float>(1.0,beta,0,0,fGemmInput,UTILITY::Clean_name(fKernelTensors.back()),UTILITY::Clean_name(fBiasTensors.back()),fFuncName+"Gemm"+std::to_string(fNumLayers)));
    function_block->AddOperator(std::move(op_gemm));
    if(fActivateFinal) {
        if (fActivationFunction == Activation::RELU) {
            std::unique_ptr<ROperator> op_relu;
            op_relu.reset(new ROperator_Relu<float>(fFuncName+"Gemm"+std::to_string(fNumLayers), fFuncName+"Relu"+std::to_string(fNumLayers)));
            function_block->AddOperator(std::move(op_relu));
        }
    }


    if(fAddlOp.size()) {
        for(auto &i:fAddlOp) {
            std::unique_ptr<ROperator> tmp(i);
            function_block->AddOperator(std::move(tmp));
        }
    }
}

void RFunction_MLP::AddLayerNormalization(int axis, float epsilon, size_t stashType, const std::string &nameX,
        const std::string &nameScale, const std::string &nameB, const std::string &nameY) {
    auto op_layerNorm = new ROperator_LayerNormalization<float>(axis, epsilon, stashType, nameX,
            nameScale, nameB, nameY, "", "");
    fAddlOp.push_back((op_layerNorm));
}

}
}
}
