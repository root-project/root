#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Clip.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// ---------------------------------------------------------------------------
// ParseClip
//
// ONNX Clip node inputs (all optional except X):
//   input(0) : X       — data tensor to clip  (required)
//   input(1) : min     — scalar lower bound    (optional)
//   input(2) : max     — scalar upper bound    (optional)
//
// ONNX Clip node output:
//   output(0): Y       — clipped output tensor
//
// If min / max inputs are absent the node may have input_size < 3.
// An absent optional input is represented in the ONNX protobuf as an
// empty string "".
// ---------------------------------------------------------------------------

ParserFuncSignature ParseClip = [](RModelParser_ONNX &parser,
                                   const onnx::NodeProto &nodeproto) {

   // ---- validate input count -------------------------------------------
   // Clip requires at least 1 input (X); min and max are optional
   if (nodeproto.input_size() < 1) {
      throw std::runtime_error(
         "TMVA::SOFIE ONNX Parser Clip op has invalid input size " +
         std::to_string(nodeproto.input_size()) + " (expected 1, 2 or 3)");
   }

   // ---- main input X must be registered --------------------------------
   if (!parser.IsRegisteredTensorType(nodeproto.input(0))) {
      throw std::runtime_error(
         "TMVA::SOFIE ONNX Parser Clip op has input tensor " +
         nodeproto.input(0) + " but its type is not yet registered");
   }

   ETensorType input_type = parser.GetTensorType(nodeproto.input(0));


   std::string minName = (nodeproto.input_size() > 1) ?  nodeproto.input(1) : "";
   std::string maxName = (nodeproto.input_size() > 2) ? nodeproto.input(2) : "";

   // ---- if min/max are provided they must match the data type ----------
   if (!minName.empty() && parser.IsRegisteredTensorType(minName)) {
      if (parser.GetTensorType(minName) != input_type) {
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser Clip op: min tensor " + minName +
            " type " + ConvertTypeToString(parser.GetTensorType(minName)) +
            " does not match input type " + ConvertTypeToString(input_type));
      }
   }
   if (!maxName.empty() && parser.IsRegisteredTensorType(maxName)) {
      if (parser.GetTensorType(maxName) != input_type) {
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser Clip op: max tensor " + maxName +
            " type " + ConvertTypeToString(parser.GetTensorType(maxName)) +
            " does not match input type " + ConvertTypeToString(input_type));
      }
   }

   // ---- build the operator ---------------------------------------------
   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Clip<float>(
         nodeproto.input(0), output_name, minName, maxName));
      break;
   case ETensorType::DOUBLE:
      op.reset(new ROperator_Clip<double>(
         nodeproto.input(0), output_name, minName, maxName));
      break;
   case ETensorType::INT32:
      op.reset(new ROperator_Clip<int32_t>(
         nodeproto.input(0), output_name, minName, maxName));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_Clip<int64_t>(
         nodeproto.input(0), output_name, minName, maxName));
      break;
   default:
      throw std::runtime_error(
         "TMVA::SOFIE - Unsupported - Clip Operator does not yet support "
         "input type " + ConvertTypeToString(input_type));
   }

   // ---- register output tensor type ------------------------------------
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
