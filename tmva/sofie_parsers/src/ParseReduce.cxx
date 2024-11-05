#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Reduce.hxx"
#include "onnx_proto3.pb.h"
#include <stdexcept>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <EReduceOpMode Op>
std::unique_ptr<ROperator> ParseReduce(RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type;

   EReduceOpMode op_mode = InvalidReduceOp;

   if (nodeproto.op_type() == "ReduceMean")
      op_mode = ReduceMean;
   else if (nodeproto.op_type() == "ReduceSumsquare")
      op_mode = ReduceSumsquare;
   else if (nodeproto.op_type() == "ReduceProd")
      op_mode = ReduceProd;
   else if (nodeproto.op_type() == "ReduceSum")
      op_mode = ReduceSum;

   if (op_mode == InvalidReduceOp) {
      throw std::runtime_error("TMVA::SOFIE - Reduce op mode not supported.");
   }

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Reduce  op has input tensor" + input_name +
                               " but its type is not yet registered");
   }
   //in the latest version of ONNX axis is not an attribute but an input
   std::string axes_name;
   if (nodeproto.input_size() > 1) {
      axes_name = nodeproto.input(1);
      if (!parser.IsRegisteredTensorType(axes_name)) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Reduce  op has input tensor" + axes_name +
                               " but its type is not yet registered");
      }
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   int attr_keepdims = 1;
   std::vector<int64_t> attr_axes;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "keepdims")
         attr_keepdims = nodeproto.attribute(i).i();
      if (attribute_name == "axes") {
         attr_axes =
            std::vector<int64_t>({nodeproto.attribute(i).ints().begin(), nodeproto.attribute(i).ints().end()});
      }
   }
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Reduce<float, Op>(attr_keepdims, attr_axes, input_name, axes_name, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Reduce Operator does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }
   return op;
}

// Parse ReduceMean
ParserFuncSignature ParseReduceMean = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseReduce<EReduceOpMode::ReduceMean>(parser, nodeproto);
};

// Parse ReduceSumsquare
ParserFuncSignature ParseReduceSumsquare = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseReduce<EReduceOpMode::ReduceSumsquare>(parser, nodeproto);
};

// Parse ReduceProd
ParserFuncSignature ParseReduceProd = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseReduce<EReduceOpMode::ReduceProd>(parser, nodeproto);
};

// Parse ReduceSum
ParserFuncSignature ParseReduceSum = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseReduce<EReduceOpMode::ReduceSum>(parser, nodeproto);
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
