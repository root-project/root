// Author: Christian Sonnabend, 2026
// For the licensing terms see $ROOTSYS/LICENSE.
#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/RGPUModel.hxx"
#include "TMVA/ROperator.hxx"
#include "onnx.hxx"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace TMVA::Experimental::SOFIE {
namespace {
float HalfToFloat(std::uint16_t bits)
{
   const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
   float value = exponent == 31
                    ? (fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                    : std::ldexp(float(exponent ? fraction + 1024 : fraction), exponent ? exponent - 25 : -24);
   return bits & 0x8000 ? -value : value;
}
std::vector<float> ReadWeights(const onnx::TensorProto &tensor, int type)
{
   if (tensor.data_type() != type || tensor.data_location() != onnx::TensorProto::DEFAULT ||
       !tensor.external_data().empty())
      throw std::runtime_error("SOFIE GPU: weights must be embedded and match the input precision");
   std::size_t count = 1;
   for (int i = 0; i < tensor.dims_size(); ++i) {
      const auto dim = tensor.dims(i);
      if (dim <= 0 || std::size_t(dim) > std::numeric_limits<std::size_t>::max() / count)
         throw std::runtime_error("SOFIE GPU: invalid initializer shape");
      count *= dim;
   }
   const auto &raw = tensor.raw_data();
   const std::size_t width = type == onnx::TensorProto::FLOAT ? 4 : 2;
   if (raw.empty()) {
      if (type == onnx::TensorProto::FLOAT && tensor.float_data().size() == count)
         return tensor.float_data();
      if (type == onnx::TensorProto::FLOAT16 && tensor.int32_data().size() == count) {
         std::vector<float> values;
         for (auto bits : tensor.int32_data())
            values.push_back(HalfToFloat(std::uint16_t(bits)));
         return values;
      }
   }
   if (raw.size() % width || raw.size() / width != count)
      throw std::runtime_error("SOFIE GPU: invalid initializer data length");
   std::vector<float> values(count);
   for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t bits = 0;
      for (std::size_t b = 0; b < width; ++b)
         bits |= std::uint32_t(static_cast<unsigned char>(raw[i * width + b])) << (8 * b);
      if (width == 2)
         values[i] = HalfToFloat(std::uint16_t(bits));
      else
         std::memcpy(&values[i], &bits, 4);
   }
   return values;
}
} // namespace
RGPUModel RModelParser_ONNX::ParseGPU(const std::string &filename)
{
   std::ifstream file(filename, std::ios::binary);
   if (!file)
      throw std::runtime_error("SOFIE GPU: cannot open " + filename);
   return ParseGPU(file);
}
RGPUModel RModelParser_ONNX::ParseGPU(std::istream &input)
{
   auto proto = LoadModel(input);
   if (!proto)
      throw std::runtime_error("SOFIE GPU: malformed ONNX input");
   const auto &graph = proto->graph();
   if (graph.input_size() != 1 || graph.output_size() != 1 || graph.node_size() == 0)
      throw std::runtime_error("SOFIE GPU: expected one input, one output and a nonempty sequential graph");
   const auto &in = graph.input(0).type().tensor_type(), &out = graph.output(0).type().tensor_type();
   const int type = in.elem_type();
   if ((type != onnx::TensorProto::FLOAT && type != onnx::TensorProto::FLOAT16) || out.elem_type() != type ||
       in.shape().dim_size() != 2 || out.shape().dim_size() != 2 || in.shape().dim(1).dim_value() <= 0)
      throw std::runtime_error("SOFIE GPU: expected matching FP32/FP16 rank-two tensors");
   const auto &batch = in.shape().dim(0), &outputBatch = out.shape().dim(0);
   if (batch.value_case() != outputBatch.value_case() || batch.dim_value() != outputBatch.dim_value() ||
       batch.dim_param() != outputBatch.dim_param() || (batch.dim_param().empty() && batch.dim_value() <= 0))
      throw std::runtime_error("SOFIE GPU: invalid or inconsistent batch dimension");
   fTensorTypeMap.clear();
   fFusedOperators.clear();
   RModel model;
   // GPU tensors retain their requested precision; operator shape inference uses promoted FP32 values.
   model.AddInputTensorInfo(graph.input(0).name(), ETensorType::FLOAT,
                            std::vector<size_t>{1, size_t(in.shape().dim(1).dim_value())});
   model.AddInputTensorName(graph.input(0).name());
   RegisterTensorType(graph.input(0).name(), ETensorType::FLOAT);
   std::unordered_set<std::string> names{graph.input(0).name()};
   for (int i = 0; i < graph.initializer_size(); ++i) {
      const auto &tensor = graph.initializer(i);
      if (!names.insert(tensor.name()).second)
         throw std::runtime_error("SOFIE GPU: duplicate initializer name");
      auto values = ReadWeights(tensor, type);
      std::vector<size_t> shape;
      for (int d = 0; d < tensor.dims_size(); ++d)
         shape.push_back(size_t(tensor.dims(d)));
      model.AddInitializedTensor(tensor.name(), ETensorType::FLOAT, shape, values.data());
      RegisterTensorType(tensor.name(), ETensorType::FLOAT);
   }
   std::string current = graph.input(0).name();
   std::vector<size_t> nodes;
   for (int i = 0; i < graph.node_size(); ++i)
      nodes.push_back(size_t(i));
   for (size_t i = 0; i < nodes.size(); ++i) {
      const auto &node = graph.node(int(i));
      if ((!node.domain().empty() && node.domain() != "ai.onnx") || node.output_size() != 1 || node.input_size() < 1 ||
          node.input(0) != current || !names.insert(node.output(0)).second)
         throw std::runtime_error("SOFIE GPU: unsupported graph connectivity/domain at " + node.name());
      if (node.op_type() == "Gemm") {
         if (node.input_size() < 2 || node.input_size() > 3)
            throw std::runtime_error("SOFIE GPU: invalid Gemm input count");
         std::unordered_set<std::string> attrs;
         for (int a = 0; a < node.attribute_size(); ++a) {
            const auto &attr = node.attribute(a);
            const bool scalar = (attr.name() == "alpha" || attr.name() == "beta") &&
                                attr.type() == onnx::AttributeProto::FLOAT;
            const bool transpose = (attr.name() == "transA" || attr.name() == "transB") &&
                                   attr.type() == onnx::AttributeProto::INT;
            if (!attrs.insert(attr.name()).second || (!scalar && !transpose))
               throw std::runtime_error("SOFIE GPU: unsupported or duplicate Gemm attribute " + attr.name());
         }
      } else if (node.op_type() == "Relu" && (node.input_size() != 1 || node.attribute_size())) {
         throw std::runtime_error("SOFIE GPU: invalid Relu inputs or attributes");
      }
      auto op = ParseOperator(i, graph, nodes, {});
      if (!op)
         throw std::runtime_error("SOFIE GPU: unexpected operator fusion");
      model.AddOperator(std::move(op));
      current = node.output(0);
   }
   model.AddOutputTensorNameList({graph.output(0).name()});
   auto gpu = model.MakeGPUModel(type == onnx::TensorProto::FLOAT ? RGPUModel::Precision::Float32
                                                                  : RGPUModel::Precision::Float16,
                                 batch.dim_param().empty() ? batch.dim_value() : 0);
   if (current != graph.output(0).name() || out.shape().dim(1).dim_value() != std::int64_t(gpu.OutputSize()))
      throw std::runtime_error("SOFIE GPU: inconsistent output shape");
   return gpu;
}
} // namespace TMVA::Experimental::SOFIE
