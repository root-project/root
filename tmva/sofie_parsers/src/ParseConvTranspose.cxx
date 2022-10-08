#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_ConvTranspose.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseConvTranspose = [](RModelParser_ONNX &model,
                                            const onnx::NodeProto &nodeproto) -> std::unique_ptr<ROperator> {
   auto inputName = nodeproto.input(0);
   if (!model.IsRegisteredTensorType(inputName)) {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ConvTranspose op has input tensor " + inputName +
                               " but its type is not yet registered");
   }
   ETensorType inputType = model.GetTensorType(inputName);

   std::string autoPad = "NOTSET";
   std::vector<size_t> dilations;
   size_t group = 0;
   std::vector<size_t> kernelShape;
   std::vector<size_t> outputPadding;
   std::vector<size_t> outputShape;
   std::vector<size_t> pads;
   std::vector<size_t> strides;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attributeName = nodeproto.attribute(i).name();
      if (attributeName == "auto_pad") {
         autoPad = nodeproto.attribute(i).s();
      } else if (attributeName == "dilations") {
         dilations = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attributeName == "group") {
         group = nodeproto.attribute(i).i();
      } else if (attributeName == "kernel_shape") {
         kernelShape =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attributeName == "output_padding") {
         outputPadding =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attributeName == "output_shape") {
         outputShape =
            std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attributeName == "pads") {
         pads = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else if (attributeName == "strides") {
         strides = std::vector<size_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      } else {
         std::cout << "TMVA::SOFIE Warning - Model Loading - Attribute " << attributeName << " in OperatorNode "
                   << nodeproto.name() << " is not defined in ONNX IR and not applied!\n";
      }
   }

   std::string nameW = nodeproto.input(1);
   std::string nameBias;
   if (nodeproto.input_size() > 2) {
      nameBias = nodeproto.input(2);
   }
   std::string outputName = nodeproto.output(0);

   std::unique_ptr<ROperator> op;
   switch (inputType) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_ConvTranspose<float>(autoPad, dilations, group, kernelShape, outputPadding, outputShape,
                                                  pads, strides, inputName, nameW, nameBias, outputName));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator ConvTranspose does not yet support input type " +
                               std::to_string(static_cast<int>(inputType)));
   }

   if (!model.IsRegisteredTensorType(outputName)) {
      model.RegisterTensorType(outputName, inputType);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
