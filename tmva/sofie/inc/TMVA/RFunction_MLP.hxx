#ifndef TMVA_SOFIE_RFUNCTION_MLP
#define TMVA_SOFIE_RFUNCTION_MLP

#include "TMVA/RFunction.hxx"

#include <vector>


namespace TMVA {
namespace Experimental {
namespace SOFIE {

enum class Activation {
    RELU = 0x0,
    Invalid = 0x1,
};

class RFunction_MLP: public RFunction_Update {
private:
    Int_t fNumLayers;           // Number of Layers in MLP
    Activation fActivationFunction;
    bool  fActivateFinal;       // if True, fActivationFunction is applied as the activation for the last layer
    std::vector<std::string> fKernelTensors;
    std::vector<std::string> fBiasTensors;

public:
    virtual ~RFunction_MLP() {}
    RFunction_MLP(FunctionTarget target, Int_t numLayers, Activation activation_function=Activation::RELU, bool activate_final=false, GraphType gType=GraphType::GNN);

    void Initialize() override;

    void AddLayerNormalization(int axis, float epsilon, size_t stashType, const std::string &nameX,
                               const std::string &nameScale, const std::string &nameB, const std::string &nameY) override;

    void AddInitializedTensors(const std::vector<std::vector<std::string>>& initialized_tensors) override {
        fKernelTensors = initialized_tensors[0];
        fBiasTensors   = initialized_tensors[1];
    }
};

} // SOFIE
} // Experimental
} // TMVA

#endif //TMVA_SOFIE_RFUNCTION_MLP
