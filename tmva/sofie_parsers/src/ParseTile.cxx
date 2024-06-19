#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Tile.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseTile = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
    // Make Tile operator
    ETensorType input_type = ETensorType::UNDEFINED;

    // Extract input data tensor name and repetitions tensor name
    std::string input_name = nodeproto.input(0);
    std::string repetitions_name = nodeproto.input(1);

    // Check if input tensor type is registered
    if (parser.IsRegisteredTensorType(input_name)) {
        input_type = parser.GetTensorType(input_name);
    } else {
        throw std::runtime_error("TMVA::SOFIE ONNX Parser Tile op has input tensor " + input_name +
                                 " but its type is not yet registered");
    }

    // Extract output tensor name
    std::string output_name = nodeproto.output(0);

    std::unique_ptr<ROperator> op;

    switch (input_type) {
        case ETensorType::FLOAT:
            // Create Tile operator instance
            op.reset(new ROperator_Tile<float>(input_name, repetitions_name, output_name));
            break;
        // Add support for other data types if needed
        default:
            throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Tile does not yet support input type " +
                                     std::to_string(static_cast<int>(input_type)));
    }

    // Register output tensor type
    if (!parser.IsRegisteredTensorType(output_name)) {
        parser.RegisterTensorType(output_name, input_type);
    }

    return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA