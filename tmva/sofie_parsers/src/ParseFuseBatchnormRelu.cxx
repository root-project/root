#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_BatchNormalization.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuseFuncSignature ParseFuseBatchnormRelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &batchnormnode,
                                                const onnx::NodeProto &relunode) {
        ETensorType input_type;

        auto input_name = batchnormnode.input(0);
        if (parser.IsRegisteredTensorType(input_name)) {
            input_type = parser.GetTensorType(input_name);
        } else {
            throw std::runtime_error("TMVA::SOFIE ONNX Parser BatchNorm op has input tensor " + input_name +
                                    " but its type is not yet registered");
        }
        
        std::unique_ptr<ROperator> op;
        std::string output_name = relunode.output(0);
        float fepsilon = 1e-05;
        float fmomentum = 0.9;
        std::size_t ftraining_mode = 0;
        
        switch (input_type) {
        case ETensorType::FLOAT:
            if (batchnormnode.input_size() == 5) {
                op.reset(new ROperator_BatchNormalization<float>(fepsilon, fmomentum, ftraining_mode, batchnormnode.input(0),
                                                                batchnormnode.input(1), batchnormnode.input(2), batchnormnode.input(3),
                                                                batchnormnode.input(4), output_name, EActivationType::RELU));
            }
            break;
        default:
            throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator BatchNorm does not yet support input type " +
                                    std::to_string(static_cast<int>(input_type)));
        }
        
        if (!parser.IsRegisteredTensorType(output_name)) {
            parser.RegisterTensorType(output_name, input_type);
        }
        
        return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
