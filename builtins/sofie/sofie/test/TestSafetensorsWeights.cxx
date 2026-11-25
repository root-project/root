// Tests for the safetensors weight payload of SOFIE, written to a file or
// handed to a session directly in memory. A tiny Mul model is written by hand
// as ONNX protobuf wire format, so the tests need neither the onnx Python
// package nor a protobuf dependency. Inference runs through the generated
// header compiled into a small program, which exercises the same code path
// users see.

#include <SOFIE/RModel.hxx>
#include <SOFIE/RModelParser_ONNX.hxx>

#include "driver_runner.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace SOFIE;

namespace {

// --- minimal protobuf wire-format writers (same encoding as the ONNX spec) --

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

// TensorProto for a float tensor with inline little-endian raw_data
std::string
InlineFloatTensor(const std::string &name, const std::vector<std::uint64_t> &shape, const std::vector<float> &values)
{
   std::string out;
   for (std::uint64_t d : shape)
      AppendVarintField(out, 1, d); // dims
   AppendVarintField(out, 2, 1);    // data_type: FLOAT
   AppendBytesField(out, 8, name);  // name
   std::string rawData;
   rawData.reserve(values.size() * sizeof(float));
   for (float value : values) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int i = 0; i < 4; ++i)
         rawData.push_back(char((bits >> (8 * i)) & 0xff));
   }
   AppendBytesField(out, 9, rawData); // raw_data
   return out;
}

// ValueInfoProto {name, type{tensor_type{elem_type: FLOAT, shape{dim_value...}}}}
std::string FloatValueInfo(const std::string &name, const std::vector<std::uint64_t> &shape)
{
   std::string shapeProto;
   for (std::uint64_t d : shape) {
      std::string dim;
      AppendVarintField(dim, 1, d); // dim_value
      AppendBytesField(shapeProto, 1, dim);
   }
   std::string tensorType;
   AppendVarintField(tensorType, 1, 1); // elem_type: FLOAT
   AppendBytesField(tensorType, 2, shapeProto);
   std::string type;
   AppendBytesField(type, 1, tensorType); // tensor_type
   std::string out;
   AppendBytesField(out, 1, name);
   AppendBytesField(out, 2, type);
   return out;
}

// ModelProto with a single "Y = Mul(X, W)" node, X a graph input and W an
// initializer holding the given values identically shaped like X.
void WriteMulModelFile(const std::string &fileName, const std::vector<std::uint64_t> &shape,
                       const std::vector<float> &weights)
{
   std::string node;
   AppendBytesField(node, 1, "X");   // input
   AppendBytesField(node, 1, "W");   // input
   AppendBytesField(node, 2, "Y");   // output
   AppendBytesField(node, 4, "Mul"); // op_type

   std::string graph;
   AppendBytesField(graph, 1, node);                                   // node
   AppendBytesField(graph, 2, "test_graph");                           // name
   AppendBytesField(graph, 5, InlineFloatTensor("W", shape, weights)); // initializer
   AppendBytesField(graph, 11, FloatValueInfo("X", shape));            // input
   AppendBytesField(graph, 12, FloatValueInfo("Y", shape));            // output

   std::string model;
   AppendVarintField(model, 1, 8);    // ir_version
   AppendBytesField(model, 7, graph); // graph

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   ASSERT_TRUE(file.good());
}

// ModelProto with a single "Z = Mul(X, Y)" node where both X and Y are graph
// inputs: the model has no weight tensors at all, exercising the codegen path
// where the safetensors helpers are not needed for real.
void WriteTwoInputMulModelFile(const std::string &fileName, const std::vector<std::uint64_t> &shape)
{
   std::string node;
   AppendBytesField(node, 1, "X");   // input
   AppendBytesField(node, 1, "Y");   // input
   AppendBytesField(node, 2, "Z");   // output
   AppendBytesField(node, 4, "Mul"); // op_type

   std::string graph;
   AppendBytesField(graph, 1, node);                     // node
   AppendBytesField(graph, 2, "test_graph");             // name
   AppendBytesField(graph, 11, FloatValueInfo("X", shape)); // input
   AppendBytesField(graph, 11, FloatValueInfo("Y", shape)); // input
   AppendBytesField(graph, 12, FloatValueInfo("Z", shape)); // output

   std::string model;
   AppendVarintField(model, 1, 8);    // ir_version
   AppendBytesField(model, 7, graph); // graph

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   ASSERT_TRUE(file.good());
}

// Parse the given model, generate inference code, and write the header (and
// weight file) next to the test binary. The returned name is the
// SOFIE_<name> namespace of the header.
std::string GenerateModel(const std::string &modelFileName)
{
   RModelParser_ONNX parser;
   RModel model = parser.Parse(modelFileName);
   model.Generate(Options::kSafetensorsWeightFile);
   model.OutputGenerated();
   return model.GetName();
}

// Compile and run a program that includes the generated header of a model, with
// the given statements as its body. The statements can write results with
// sofie_test::write(std::cout, ...).
DriverResult RunProgram(const std::string &modelName, const std::string &body)
{
   std::stringstream driver;
   driver << "#include \"" << modelName << ".hxx\"\n"
          << "#include \"driver_support.hxx\"\n"
          << "#include <fstream>\n#include <iostream>\n#include <iterator>\n#include <string>\n#include <vector>\n"
          << "namespace Model = SOFIE_" << modelName << ";\n"
          << "int main() {\n   try {\n"
          << body << "\n"
          << "   } catch (std::exception const &e) {\n"
          << "      std::cerr << \"exception: \" << e.what() << std::endl;\n"
          << "      return 1;\n"
          << "   }\n   return 0;\n}\n";
   return compileAndRunDriver(driver.str());
}

// Run a program and read back the result of type T it writes
template <class T>
T RunProgramForResult(const std::string &modelName, const std::string &body)
{
   DriverResult result = RunProgram(modelName, body);
   if (!checkDriver(result, "model " + modelName))
      return T{};
   std::istringstream in(result.output);
   return sofie_test::Reader{in}.read<T>();
}

std::string FloatVectorDeclaration(const std::string &name, const std::vector<float> &values)
{
   std::stringstream out;
   declareInput(out, name, values);
   return out.str();
}

// Compile the generated model header
void DeclareModel(const std::string &modelName)
{
   DriverResult result = RunProgram(modelName, "");
   ASSERT_TRUE(result.compiled) << "failed to compile header " << modelName << ".hxx:\n" << result.compileLog;
}

// Run inference of a model on the given input tensor
std::vector<float>
RunModel(const std::string &modelName, const std::string &dataFileName, const std::vector<float> &input)
{
   return RunProgramForResult<std::vector<float>>(modelName,
                                                  FloatVectorDeclaration("input", input) + "Model::Session session(\"" +
                                                     dataFileName +
                                                     "\");\n"
                                                     "std::vector<float> output = session.infer(input.data());\n"
                                                     "sofie_test::write(std::cout, output);");
}

// Evaluate the session constructor, reporting whether it throws
bool SessionConstructorThrows(const std::string &modelName, const std::string &dataFileName)
{
   return RunProgramForResult<bool>(modelName, "bool throws = false;\n"
                                               "try { Model::Session s(\"" +
                                                  dataFileName +
                                                  "\"); } catch (const std::exception &) { throws = true; }\n"
                                                  "sofie_test::write(std::cout, throws);");
}

// Read a file fully into memory
std::string ReadWholeFile(const std::string &fileName)
{
   std::ifstream f(fileName, std::ios::binary);
   if (!f)
      throw std::runtime_error("cannot open " + fileName);
   return std::string{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Write a weights blob to a file, from where the program reads it back in memory
std::string WriteBlobFile(const std::string &blob)
{
   // The tests can run in parallel in the same directory, so the file name is unique to the test
   static int counter = 0;
   const std::string fileName = std::string("safetensors_blob_") +
                                ::testing::UnitTest::GetInstance()->current_test_info()->name() + "_" +
                                std::to_string(counter++) + ".bin";
   std::ofstream f(fileName, std::ios::binary);
   f.write(blob.data(), blob.size());
   return fileName;
}

// Statements that read a weights blob from a file into a std::string called blob
std::string ReadBlobStatements(const std::string &blobFileName)
{
   return "std::ifstream blobFile(\"" + blobFileName +
          "\", std::ios::binary);\n"
          "std::string blob{std::istreambuf_iterator<char>(blobFile), std::istreambuf_iterator<char>()};\n";
}

// Run inference with a session constructed from a weights blob held in memory
std::vector<float>
RunModelFromBlob(const std::string &modelName, const std::string &blob, const std::vector<float> &input)
{
   return RunProgramForResult<std::vector<float>>(
      modelName, ReadBlobStatements(WriteBlobFile(blob)) + FloatVectorDeclaration("input", input) +
                    "Model::Session session(Model::SafetensorsBlob{blob.data(), blob.size()});\n"
                    "std::vector<float> output = session.infer(input.data());\n"
                    "sofie_test::write(std::cout, output);");
}

// Run inference through a blob session on a model with two float inputs
std::vector<float> RunTwoInputModelFromBlob(const std::string &modelName, const std::string &blob,
                                            std::vector<float> &inputX, std::vector<float> &inputY)
{
   return RunProgramForResult<std::vector<float>>(
      modelName, ReadBlobStatements(WriteBlobFile(blob)) + FloatVectorDeclaration("inputX", inputX) +
                    FloatVectorDeclaration("inputY", inputY) +
                    "Model::Session session(Model::SafetensorsBlob{blob.data(), blob.size()});\n"
                    "std::vector<float> output = session.infer(inputX.data(), inputY.data());\n"
                    "sofie_test::write(std::cout, output);");
}

// Build a raw safetensors payload from a JSON header and byte payload
std::string MakeSafetensorsPayload(const std::string &header, const std::string &payload)
{
   std::string out;
   for (int i = 0; i < 8; ++i)
      out.push_back(char((header.size() >> (8 * i)) & 0xff));
   out += header;
   out += payload;
   return out;
}

// Evaluate a blob session constructor, reporting whether it throws
bool BlobSessionConstructorThrows(const std::string &modelName, const std::string &blob)
{
   return RunProgramForResult<bool>(modelName,
                                    ReadBlobStatements(WriteBlobFile(blob)) +
                                       "bool throws = false;\n"
                                       "try { Model::Session s(Model::SafetensorsBlob{blob.data(), blob.size()}); }\n"
                                       "catch (const std::exception &) { throws = true; }\n"
                                       "sofie_test::write(std::cout, throws);");
}

// Values covering the corner cases the old text weight format could not
// represent exactly (a subnormal value and an infinity); the binary format
// round-trips them bit-exactly
const std::vector<float> kWeights{1.5f, -2.25f, 1.0e-42f, std::numeric_limits<float>::infinity(), 0.0f, 123456.75f};
const std::vector<float> kInput{0.5f, -1.0f, 3.0f, 1.0f, -8.0f, 0.25f};

std::vector<float> ExpectedOutput()
{
   std::vector<float> out(kWeights.size());
   for (std::size_t i = 0; i < out.size(); ++i)
      out[i] = kWeights[i] * kInput[i];
   return out;
}

} // namespace

// The safetensors payloads are raw little-endian tensor data; the writer and
// reader refuse to work on big-endian hosts (see the endianness guards in
// RModel::WriteInitializedTensorsToStream and the generated SafetensorsReader).
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define SOFIE_SKIP_ON_BIG_ENDIAN GTEST_SKIP() << "safetensors weights are little-endian only"
constexpr bool kRunsOnLittleEndian = false;
#else
#define SOFIE_SKIP_ON_BIG_ENDIAN
constexpr bool kRunsOnLittleEndian = true;
#endif

// Each test runs in a directory of its own, where the default model is
// generated, so that the tests don't depend on each other: they can run in any
// order, or at the same time as separate tests of ctest.
class SOFIESafetensors : public ::testing::Test {
protected:
   void SetUp() override
   {
      if (!kRunsOnLittleEndian)
         return;
      const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
      const std::filesystem::path dir = std::filesystem::path("safetensors_tests") / info->name();
      std::filesystem::create_directories(dir);
      std::filesystem::current_path(std::filesystem::absolute(dir));
      WriteMulModelFile("safetensors_mul.onnx", {2, 3}, kWeights);
      GenerateModel("safetensors_mul.onnx");
   }
};

// The safetensors writer emits a well-formed file: 8-byte little-endian JSON
// header size, the JSON header describing each tensor, and the raw payloads
TEST_F(SOFIESafetensors, FileLayoutAndPayload)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteMulModelFile("safetensors_mul.onnx", {2, 3}, kWeights);
   const std::string name = GenerateModel("safetensors_mul.onnx");
   ASSERT_EQ(name, "safetensors_mul");

   std::ifstream f("safetensors_mul.safetensors", std::ios::binary);
   ASSERT_TRUE(f.is_open());
   std::uint64_t headerSize = 0;
   char sizestr[8];
   f.read(sizestr, 8);
   ASSERT_TRUE(f.good());
   for (int i = 0; i < 8; ++i)
      headerSize |= std::uint64_t(static_cast<unsigned char>(sizestr[i])) << (8 * i);

   std::string headerStr(headerSize, '\0');
   f.read(headerStr.data(), headerStr.size());
   ASSERT_TRUE(f.good());
   const auto header = nlohmann::json::parse(headerStr);

   // the model has exactly one weight tensor
   ASSERT_EQ(header.size(), 1u);
   ASSERT_TRUE(header.contains("tensor_W"));
   const auto &entry = header["tensor_W"];
   EXPECT_EQ(entry["dtype"], "F32");
   EXPECT_EQ(entry["shape"], (nlohmann::json::array_t{2, 3}));
   EXPECT_EQ(entry["data_offsets"], (nlohmann::json::array_t{0, kWeights.size() * sizeof(float)}));

   // the payload follows right after the header, byte-identical to the weights
   std::vector<char> payload(kWeights.size() * sizeof(float));
   f.read(payload.data(), payload.size());
   ASSERT_TRUE(f.good());
   EXPECT_EQ(std::memcmp(payload.data(), kWeights.data(), payload.size()), 0);
   // nothing but the payload in the file
   EXPECT_EQ(f.peek(), EOF);
}

// The safetensors payloads are written in sorted tensor-name order so that
// repeated generation of the same model produces the same file
TEST_F(SOFIESafetensors, DeterministicOutput)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteMulModelFile("safetensors_mul.onnx", {2, 3}, kWeights);
   GenerateModel("safetensors_mul.onnx");
   std::ifstream first("safetensors_mul.safetensors", std::ios::binary);
   const std::string content1{std::istreambuf_iterator<char>(first), std::istreambuf_iterator<char>()};
   GenerateModel("safetensors_mul.onnx");
   std::ifstream second("safetensors_mul.safetensors", std::ios::binary);
   const std::string content2{std::istreambuf_iterator<char>(second), std::istreambuf_iterator<char>()};
   EXPECT_EQ(content1, content2);
}

// A Session built from a safetensors weight file reproduces the reference
// inference, bit-exactly
TEST_F(SOFIESafetensors, InferenceMatchesReference)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   DeclareModel("safetensors_mul");

   const std::vector<float> output = RunModel("safetensors_mul", "safetensors_mul.safetensors", kInput);
   const std::vector<float> expected = ExpectedOutput();

   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(output[i], expected[i]) << "at output index " << i;
}

// A Session built with no explicit file name finds the default
// <model name>.safetensors weight file
TEST_F(SOFIESafetensors, DefaultWeightFileName)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const auto output = RunProgramForResult<std::vector<float>>(
      "safetensors_mul", FloatVectorDeclaration("input", kInput) +
                            "Model::Session session;\n"
                            "std::vector<float> output = session.infer(input.data());\n"
                            "sofie_test::write(std::cout, output);");
   EXPECT_EQ(output, ExpectedOutput());
}

// A missing weight file is an error, not silently zeroed weights
TEST_F(SOFIESafetensors, MissingFileThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   EXPECT_TRUE(SessionConstructorThrows("safetensors_mul", "safetensors_mul_bogus.safetensors"));
}

// A tensor whose declared dtype does not match what the generated code expects
// is rejected
TEST_F(SOFIESafetensors, WrongDtypeThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::ifstream in("safetensors_mul.safetensors", std::ios::binary);
   std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
   in.close();
   // flip the F32 dtype token to F64 (same length, so the layout stays valid)
   const std::size_t pos = content.find("\"F32\"");
   ASSERT_NE(pos, std::string::npos);
   content.replace(pos, 5, "\"F64\"");
   std::ofstream out("safetensors_mul_bad_dtype.safetensors", std::ios::binary);
   out.write(content.data(), content.size());
   out.close();

   EXPECT_TRUE(SessionConstructorThrows("safetensors_mul", "safetensors_mul_bad_dtype.safetensors"));
}

// The embedded JSON parser accepts safetensors files written by other tools:
// arbitrary whitespace and field order, a __metadata__ block, string escapes
TEST_F(SOFIESafetensors, ThirdPartyFileLayout)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   // A hand-crafted header with pretty-printing, a metadata block carrying an
   // escaped string, and the fields in a different order than SOFIE emits them
   const std::string headerStr = R"({
   "__metadata__": {
      "generator": "hand-written \"test\" file\ngenerated for SOFIE",
      "format": "pt"
   },
   "tensor_W":
   {
      "data_offsets" : [ 0, 24 ],
      "shape" : [ 3, 2 ],
      "dtype" : "F32"
   }
})";
   std::ofstream out("safetensors_mul_thirdparty.safetensors", std::ios::binary);
   // 8-byte little-endian header length
   for (int i = 0; i < 8; ++i)
      out.put(char((headerStr.size() >> (8 * i)) & 0xff));
   out << headerStr;
   for (float value : kWeights) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int i = 0; i < 4; ++i)
         out.put(char((bits >> (8 * i)) & 0xff));
   }
   out.close();

   const std::vector<float> output = RunModel("safetensors_mul", "safetensors_mul_thirdparty.safetensors", kInput);
   EXPECT_EQ(output, ExpectedOutput());
}

// A syntactically invalid JSON header is rejected
TEST_F(SOFIESafetensors, MalformedJsonThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::string headerStr = R"({"tensor_W": {"dtype": "F32", "data_offsets": [0, 24},)"; // note the broken bracket
   std::ofstream out("safetensors_mul_malformed.safetensors", std::ios::binary);
   for (int i = 0; i < 8; ++i)
      out.put(char((headerStr.size() >> (8 * i)) & 0xff));
   out << headerStr;
   out.close();

   EXPECT_TRUE(SessionConstructorThrows("safetensors_mul", "safetensors_mul_malformed.safetensors"));
}

// Sessions can be constructed from a safetensors payload held in memory,
// without any weight file involved at all
TEST_F(SOFIESafetensors, InferenceFromMemoryBlob)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string blob = ReadWholeFile("safetensors_mul.safetensors");
   const std::vector<float> output = RunModelFromBlob("safetensors_mul", blob, kInput);
   EXPECT_EQ(output, ExpectedOutput());
}

// The buffer produced by WriteInitializedTensorsToBuffer is identical to the
// safetensors file content
TEST_F(SOFIESafetensors, InMemoryWriterMatchesFile)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteMulModelFile("safetensors_mul.onnx", {2, 3}, kWeights);
   RModelParser_ONNX parser;
   RModel model = parser.Parse("safetensors_mul.onnx");
   model.Generate(Options::kSafetensorsWeightFile);
   EXPECT_EQ(model.WriteInitializedTensorsToBuffer(), ReadWholeFile("safetensors_mul.safetensors"));
}

// A truncated weights blob is rejected at session construction
TEST_F(SOFIESafetensors, TruncatedBlobThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string full = ReadWholeFile("safetensors_mul.safetensors");
   ASSERT_GT(full.size(), 10u);
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", full.substr(0, full.size() - 10)));
}

// A model without weight tensors still builds a session from an in-memory
// blob (which is simply ignored), so that both construction modes exist
// whatever the model
TEST_F(SOFIESafetensors, WeightlessModelFromBlob)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteTwoInputMulModelFile("safetensors_weightless.onnx", {2, 3});
   ASSERT_EQ(GenerateModel("safetensors_weightless.onnx"), "safetensors_weightless");
   DeclareModel("safetensors_weightless");
   // an empty blob is fine: no weights are read from it
   std::vector<float> inX{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
   std::vector<float> inY{2.0f, -1.0f, 0.5f, 3.0f, -4.0f, 7.0f};
   std::vector<float> expected(inX.size());
   for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = inX[i] * inY[i];
   EXPECT_EQ(RunTwoInputModelFromBlob("safetensors_weightless", "", inX, inY), expected);
}

// Deeply nested JSON in an ignored member must not exhaust the stack
TEST_F(SOFIESafetensors, DeepNestingThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::string header = R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":)";
   header += std::string(10000, '[') + std::string(10000, ']') + "}";
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}

// The same depth of metadata that fits under the budget is accepted
TEST_F(SOFIESafetensors, ModerateNestingAccepted)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::string header = R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":)";
   header += std::string(32, '[') + std::string(32, ']') + "}";
   EXPECT_FALSE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}

// data_offsets crafted to wrap around uint64_t when added to the payload base
// must be rejected, not silently read from the header area
TEST_F(SOFIESafetensors, WrappingOffsetsThrow)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   // payload base is 8 + header size; this end offset overflows when added
   const std::string header =
      R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,18446744073709551600]}})";
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}

// offsets that are integers but with a fraction marker like 24.0 are rejected
TEST_F(SOFIESafetensors, NonIntegerOffsetsThrow)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string header = R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24.0]}})";
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}
