// Unit tests for the SOFIE ONNX parser layer (RModelParser_ONNX).
// The model files are hand-written protobuf wire format, so the tests need
// neither the onnx Python package nor a protobuf dependency.

#include <TMVA/RModel.hxx>
#include <TMVA/RModelParser_ONNX.hxx>

#include <TSystem.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace TMVA::Experimental::SOFIE;

namespace {

// --- minimal protobuf wire-format writers ----------------------------------

void AppendVarint(std::string &out, std::uint64_t v)
{
   while (v >= 0x80) {
      out.push_back(char((v & 0x7f) | 0x80));
      v >>= 7;
   }
   out.push_back(char(v));
}

void AppendVarintField(std::string &out, int field, std::uint64_t v)
{
   AppendVarint(out, std::uint64_t(field) << 3 | 0); // wire type 0: varint
   AppendVarint(out, v);
}

void AppendBytesField(std::string &out, int field, const std::string &payload)
{
   AppendVarint(out, std::uint64_t(field) << 3 | 2); // wire type 2: length-delimited
   AppendVarint(out, payload.size());
   out += payload;
}

// StringStringEntryProto
std::string StringEntry(const std::string &key, const std::string &value)
{
   std::string out;
   AppendBytesField(out, 1, key);
   AppendBytesField(out, 2, value);
   return out;
}

// TensorProto for a 1-d float tensor whose data is stored externally. An
// empty location means no "location" entry, so the parser falls back to the
// conventional <model file>.data name.
std::string
ExternalFloatTensor(const std::string &name, std::uint64_t dim, std::uint64_t offset, const std::string &location = "")
{
   std::string out;
   AppendVarintField(out, 1, dim); // dims
   AppendVarintField(out, 2, 1);   // data_type: FLOAT
   AppendBytesField(out, 8, name); // name
   if (!location.empty())
      AppendBytesField(out, 13, StringEntry("location", location)); // external_data
   AppendBytesField(out, 13, StringEntry("offset", std::to_string(offset)));
   AppendBytesField(out, 13, StringEntry("length", std::to_string(dim * sizeof(float))));
   AppendVarintField(out, 14, 1); // data_location: EXTERNAL
   return out;
}

// ModelProto holding a graph with the given initializers. The parser does
// not support graphs without nodes, so route the first tensor through an
// Identity node to the graph output.
void WriteModelFile(const std::string &fileName, const std::vector<std::string> &initializers,
                    const std::string &firstTensorName)
{
   std::string node;
   AppendBytesField(node, 1, firstTensorName); // input
   AppendBytesField(node, 2, "out");           // output
   AppendBytesField(node, 4, "Identity");      // op_type

   std::string output;
   AppendBytesField(output, 1, "out"); // name

   std::string graph;
   AppendBytesField(graph, 1, node);         // node
   AppendBytesField(graph, 2, "test_graph"); // name
   for (const std::string &tensor : initializers)
      AppendBytesField(graph, 5, tensor); // initializer
   AppendBytesField(graph, 12, output);   // output

   std::string model;
   AppendVarintField(model, 1, 8);    // ir_version
   AppendBytesField(model, 7, graph); // graph

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   ASSERT_TRUE(file.good());
}

// External weight data is little-endian per the ONNX spec, like raw_data
void WriteDataFile(const std::string &fileName, const std::vector<float> &values, std::size_t padding = 0)
{
   std::ofstream file(fileName, std::ios::binary);
   for (std::size_t i = 0; i < padding; ++i)
      file.put('\0');
   for (float value : values) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int i = 0; i < 4; ++i)
         file.put(char((bits >> (8 * i)) & 0xff));
   }
   ASSERT_TRUE(file.good());
}

} // namespace

// The "location" key of a tensor's external_data names the data file relative
// to the model directory; different tensors can point to different files.
TEST(SOFIEParser, ExternalDataLocationRelativeToModelDirectory)
{
   gSystem->mkdir("extdata_models");
   const std::vector<float> values1{1.f, 2.f, 3.f, 4.f};
   const std::vector<float> values2{-5.f, 6.5f};
   WriteDataFile("extdata_models/weights1.bin", values1, /*padding=*/8);
   WriteDataFile("extdata_models/weights2.bin", values2);
   WriteModelFile("extdata_models/modelLoc.onnx",
                  {ExternalFloatTensor("w1", values1.size(), /*offset=*/8, "weights1.bin"),
                   ExternalFloatTensor("w2", values2.size(), /*offset=*/0, "weights2.bin")},
                  "w1");

   RModelParser_ONNX parser;
   RModel model = parser.Parse("extdata_models/modelLoc.onnx");
   EXPECT_EQ(model.GetTensorData<float>("w1"), values1);
   EXPECT_EQ(model.GetTensorData<float>("w2"), values2);
}

// A tensor without a "location" entry falls back to the conventional
// <model file>.data name, resolved per Parse call: reusing one parser
// instance for several models must not read the first model's data file.
TEST(SOFIEParser, ExternalDataFileResolvedPerParsedModel)
{
   const std::vector<float> valuesA{10.f, 20.f, 30.f};
   const std::vector<float> valuesB{-1.f, -2.f, -3.f};
   WriteDataFile("extdataA.onnx.data", valuesA);
   WriteDataFile("extdataB.onnx.data", valuesB);
   WriteModelFile("extdataA.onnx", {ExternalFloatTensor("w", valuesA.size(), /*offset=*/0)}, "w");
   WriteModelFile("extdataB.onnx", {ExternalFloatTensor("w", valuesB.size(), /*offset=*/0)}, "w");

   RModelParser_ONNX parser;
   RModel modelA = parser.Parse("extdataA.onnx");
   EXPECT_EQ(modelA.GetTensorData<float>("w"), valuesA);
   RModel modelB = parser.Parse("extdataB.onnx");
   EXPECT_EQ(modelB.GetTensorData<float>("w"), valuesB);
}

// A file set with SetExternalDataFile takes precedence over the stored
// location, but only for the next Parse call.
TEST(SOFIEParser, SetExternalDataFileTakesPrecedenceOnce)
{
   const std::vector<float> valuesExplicit{5.f, 6.f};
   const std::vector<float> valuesLocation{7.f, 8.f};
   WriteDataFile("extdata_explicit.bin", valuesExplicit);
   WriteDataFile("extdata_location.bin", valuesLocation);
   WriteModelFile("extdataC.onnx",
                  {ExternalFloatTensor("w", valuesExplicit.size(), /*offset=*/0, "extdata_location.bin")}, "w");

   RModelParser_ONNX parser;
   parser.SetExternalDataFile("extdata_explicit.bin");
   RModel modelExplicit = parser.Parse("extdataC.onnx");
   EXPECT_EQ(modelExplicit.GetTensorData<float>("w"), valuesExplicit);

   RModel modelLocation = parser.Parse("extdataC.onnx");
   EXPECT_EQ(modelLocation.GetTensorData<float>("w"), valuesLocation);
}

// A data file that cannot be opened is an error, not silently zeroed weights
TEST(SOFIEParser, MissingExternalDataFileThrows)
{
   WriteModelFile("extdataD.onnx", {ExternalFloatTensor("w", 2, /*offset=*/0, "extdata_does_not_exist.bin")}, "w");

   RModelParser_ONNX parser;
   EXPECT_THROW(parser.Parse("extdataD.onnx"), std::runtime_error);
}
