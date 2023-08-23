#include "TMVA/RModel.hxx"
#include "TMVA/RFunction.hxx"


namespace TMVA {
namespace Experimental {
namespace SOFIE {



RFunction_Update::RFunction_Update(FunctionTarget target, GraphType gType): fTarget(target), fGraphType(gType) {
    switch(target) {
    case FunctionTarget::EDGES: {
        fFuncName = "edge_update";
        break;
    }
    case FunctionTarget::NODES: {
        fFuncName = "node_update";
        break;
    }
    case FunctionTarget::GLOBALS: {
        fFuncName = "global_update";
        break;
    }
    default:
        throw std::runtime_error("Invalid target for Update function");
    }
    fType = FunctionType::UPDATE;
    function_block = std::make_unique<RModel>(fFuncName);

    if(fGraphType == GraphType::GNN) {
        if(fTarget == FunctionTarget::EDGES) {
            fInputTensors = {"edge","receiver","sender","global"};
        } else if(fTarget == FunctionTarget::NODES || fTarget == FunctionTarget::GLOBALS) {
            fInputTensors = {"edge","node","global"};
        }

    } else if(fGraphType == GraphType::GraphIndependent) {
        if(fTarget == FunctionTarget::EDGES) {
            fInputTensors = {"edge"};
        } else if(fTarget == FunctionTarget::NODES) {
            fInputTensors = {"node"};
        } else {
            fInputTensors = {"global"};
        }
    }
}

void RFunction_Update::AddInputTensors(const std::vector<std::vector<std::size_t>>& fInputShape) {
    for(long unsigned int i=0; i<fInputShape.size(); ++i) {
        function_block->AddInputTensorInfo(fInputTensors[i],ETensorType::FLOAT, fInputShape[i]);
        function_block->AddInputTensorName(fInputTensors[i]);
    }
}

std::string RFunction_Update::GenerateModel(const std::string& filename, long read_pos, long block_size) {
    function_block->SetFilename(filename);
    // use batch size as block size in RModel::generate
    function_block->Generate(Options::kGNNComponent,block_size,read_pos);
    std::string modelGenerationString;
    modelGenerationString = "\n//--------- GNN_Update_Function---"+fFuncName+"\n"+function_block->ReturnGenerated();
    return modelGenerationString;
}

std::string RFunction_Update::Generate(const std::vector<std::string>& inputPtrs) {
    std::string inferFunc = fFuncName+".infer(";
    for(auto&it : inputPtrs) {
        inferFunc+=it;
        inferFunc+=",";
    }
    inferFunc.pop_back();
    inferFunc+=");";
    return inferFunc;
}


std::string RFunction_Aggregate::Generate(std::size_t num_features, const std::vector<std::string>& inputTensors) {
    std::string inferFunc = fFuncName+"("+std::to_string(num_features)+",{";
    for(auto&it : inputTensors) {
        inferFunc+=it;
        inferFunc+=",";
    }
    inferFunc.pop_back();
    inferFunc+="});";
    return inferFunc;
}




}
}
}
