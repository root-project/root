#ifndef TMVA_SOFIE_RFUNCTION
#define TMVA_SOFIE_RFUNCTION

#include "TMVA/RModel_Base.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <memory>
#include <string>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModel;


class RFunction {
protected:
    std::string fFuncName;
    FunctionType fType;
public:
    RFunction() {}
    virtual ~RFunction() {}
    FunctionType GetFunctionType() {
        return fType;
    }

    RFunction(std::string funcName, FunctionType type):
        fFuncName(UTILITY::Clean_name(funcName)),fType(type) {}

};

class RFunction_Update: public RFunction {
protected:
    std::shared_ptr<RModel> function_block;
    FunctionTarget fTarget;
    GraphType fGraphType;
    std::vector<std::string> fInputTensors;
    std::vector<ROperator*> fAddlOp;  // temporary vector to store pointer that will be moved in a unique_ptr

public:
    virtual ~RFunction_Update() {}
    RFunction_Update() {}
    RFunction_Update(FunctionTarget target, GraphType gType);

    virtual void AddInitializedTensors(const std::vector<std::vector<std::string>>&) {};
    virtual void Initialize() {};
    virtual void AddLayerNormalization(int, float, size_t, const std::string&,
                                       const std::string&, const std::string&, const std::string&) {};
    void AddInputTensors(const std::vector<std::vector<std::size_t>>& fInputShape);
    std::shared_ptr<RModel> GetFunctionBlock() {
        return function_block;
    }

    std::string GenerateModel(const std::string& filename, long read_pos=0, long block_size=1);
    std::string Generate(const std::vector<std::string>& inputPtrs);
    FunctionTarget GetFunctionTarget() {
        return fTarget;
    }
};

class RFunction_Aggregate: public RFunction {
protected:
    FunctionReducer fReducer;
public:
    virtual ~RFunction_Aggregate() {}
    RFunction_Aggregate() {}
    RFunction_Aggregate(FunctionReducer reducer): fReducer(reducer) {
        fType = FunctionType::AGGREGATE;
    }
    virtual std::string GenerateModel() = 0;
    std::string GetFunctionName() {
        return fFuncName;
    }
    FunctionReducer GetFunctionReducer() {
        return fReducer;
    }
    std::string Generate(std::size_t num_features, const std::vector<std::string>& inputTensors);
    std::string Generate(std::size_t num_features, const std::string & inputTensors);

};


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_RFUNCTION
