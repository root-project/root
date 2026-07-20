#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <TInterpreter.h>

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

/// Reference data for one model: the test inputs and the expected outputs,
/// read from the references/<Model>.ref file that generate_input_models.py
/// writes next to the generated ONNX models. Entries are keyed "input0",
/// "input1", ... (one per graph input, in graph order) and "output0", ...
/// (one per graph output).
class SofieReference {
public:
   const std::vector<float> &f32(std::string const &key) const { return at(fF32, key); }
   const std::vector<double> &f64(std::string const &key) const { return at(fF64, key); }
   const std::vector<int64_t> &i64(std::string const &key) const { return at(fI64, key); }
   const std::vector<uint8_t> &u8(std::string const &key) const { return at(fU8, key); }

   std::map<std::string, std::vector<float>> fF32;
   std::map<std::string, std::vector<double>> fF64;
   std::map<std::string, std::vector<int64_t>> fI64;
   std::map<std::string, std::vector<uint8_t>> fU8;

private:
   template <class Map>
   static const typename Map::mapped_type &at(Map const &m, std::string const &key)
   {
      auto it = m.find(key);
      if (it == m.end())
         throw std::runtime_error("no reference data entry \"" + key + "\" of this type");
      return it->second;
   }
};

inline SofieReference readReference(std::string const &modelName)
{
   const std::string path = "input_models/references/" + modelName + ".ref";
   std::ifstream in(path);
   if (!in)
      throw std::runtime_error("cannot open reference data file " + path +
                               " (it is written by the SofieGenerateModels_ONNX test)");
   SofieReference ref;
   std::string key, type;
   std::size_t count;
   while (in >> key >> type >> count) {
      bool ok = true;
      if (type == "f32") {
         auto &v = ref.fF32[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "f64") {
         auto &v = ref.fF64[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "i64") {
         auto &v = ref.fI64[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "u8") {
         auto &v = ref.fU8[key];
         v.resize(count);
         for (auto &x : v) {
            int tmp;
            ok &= bool(in >> tmp);
            x = static_cast<uint8_t>(tmp);
         }
      } else {
         ok = false;
      }
      if (!ok)
         throw std::runtime_error("malformed reference data file " + path + " at entry \"" + key + "\"");
   }
   return ref;
}

/// Element-wise |output - expected| <= tolerance
template <typename T, typename U>
void expectNear(std::vector<T> const &output, std::vector<U> const &expected, float tolerance)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i)
      EXPECT_LE(std::abs(static_cast<double>(output[i]) - static_cast<double>(expected[i])), tolerance)
         << "at output index " << i;
}

/// Element-wise |output - expected| <= tolerance, for models with several outputs
template <typename T, typename U>
void expectNear(std::vector<std::vector<T>> const &output, std::vector<std::vector<U>> const &expected, float tolerance)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i) {
      SCOPED_TRACE("output tensor " + std::to_string(i));
      expectNear(output[i], expected[i], tolerance);
   }
}

/// Element-wise exact equality
template <typename T, typename U>
void expectEqual(std::vector<T> const &output, std::vector<U> const &expected)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i)
      EXPECT_EQ(static_cast<int64_t>(output[i]), static_cast<int64_t>(expected[i])) << "at output index " << i;
}

/// Element-wise exact equality, for models with several outputs
template <typename T, typename U>
void expectEqual(std::vector<std::vector<T>> const &output, std::vector<std::vector<U>> const &expected)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i) {
      SCOPED_TRACE("output tensor " + std::to_string(i));
      expectEqual(output[i], expected[i]);
   }
}

bool includeModel(std::string const &modelName)
{
   const std::string header = modelName + modelHeaderSuffix;

   const std::string decl = R"(#include ")" + header + R"(")";

   if (gInterpreter->Declare(decl.c_str())) {
      return true;
   }

   // --- Declaration failed: dump header for debugging ---
   std::cerr << "\n[includeModel] Failed to declare model: " << modelName << '\n'
             << "[includeModel] Header file: " << header << '\n';

   std::ifstream in(header);
   if (!in) {
      std::cerr << "[includeModel] ERROR: could not open header file\n";
      return false;
   }

   std::cerr << "========== BEGIN " << header << " ==========\n";

   std::string line;
   while (std::getline(in, line)) {
      std::cerr << line << '\n';
   }

   std::cerr << "=========== END " << header << " ===========\n";

   return false;
}

template <class T>
std::string toInterpreter(T const &ptr, std::string const &className, bool toRawPointer = false)
{
   // for the integer arguments (shape values)
   if constexpr (std::is_same_v<T, int> || std::is_same_v<T, size_t>) {
      return std::to_string(ptr);
   }
   // for the data arguments
   std::string out =
      TString::Format("reinterpret_cast<%s*>(0x%zx)", className.c_str(), reinterpret_cast<std::size_t>(&ptr)).Data();
   if (toRawPointer) {
      out += "->data()";
   }
   return out;
}

// Output type names without commas in the name, to be used in macro calls
using TupleFloatInt64_t = std::tuple<std::vector<float>, std::vector<int64_t>>;

// Helper: map C++ type -> string used in interpreter
template <typename T>
constexpr const char *type_name()
{
   if constexpr (std::is_same_v<T, int>)
      return "int";
   else if constexpr (std::is_same_v<T, size_t>)
      return "size_t";
   else if constexpr (std::is_same_v<T, std::vector<float>>)
      return "std::vector<float>";
   else if constexpr (std::is_same_v<T, std::vector<int>>)
      return "std::vector<int>";
   else if constexpr (std::is_same_v<T, std::vector<int64_t>>)
      return "std::vector<int64_t>";
   else if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
      return "std::vector<uint8_t>";
   else
      static_assert(!sizeof(T), "Input type not supported");
}

template <typename T>
void emitInput(std::stringstream &cmd, const T &input, bool &first)
{
   if (!first)
      cmd << ", ";
   first = false;

   cmd << toInterpreter(input, type_name<T>(), true);
}

template <typename OutputType_t, typename... Ts>
OutputType_t
runModel(std::string outputTypeName, std::string const &modelName, std::string sessionArgs, Ts const &...inputs)
{
   OutputType_t output;

   // The interpreter doesn't know about our aliases, to we convert them back
   if (outputTypeName == "TupleFloatInt64_t") {
      outputTypeName = "std::tuple<std::vector<float>, std::vector<int64_t>>";
   }

   std::stringstream cmd;

   if (sessionArgs.empty()) {
      sessionArgs = R"(")" + modelName + modelDataSuffix + R"(")";
   }

   if (sessionArgs != "NO_SESSION") {
      cmd << R"(
   TMVA_SOFIE_)"
          << modelName << R"(::Session s()" << sessionArgs << R"();
   )" << outputTypeName;

      cmd << R"( output = s.infer()";
   } else {
      cmd << outputTypeName << R"( output = TMVA_SOFIE_)" << modelName << R"(::infer()";
   }

   // Emit all inputs to s.infer(...)
   if constexpr (sizeof...(Ts) > 0) {
      bool first = true;
      (emitInput<Ts>(cmd, inputs, first), ...);
   }

   cmd << R"();
   std::swap(output, *)"
       << toInterpreter(output, outputTypeName) << R"();
   )";

   gInterpreter->ProcessLine(cmd.str().c_str());

   return output;
}

#define ASSERT_INCLUDE_AND_RUN_0(OutputType, modelLiteral, ...)                       \
   const std::string _modelName = (modelLiteral);                                     \
   ASSERT_TRUE(includeModel(_modelName)) << "Failed to include model " << _modelName; \
   auto output = runModel<OutputType>(#OutputType, _modelName, "");

#define ASSERT_INCLUDE_AND_RUN(OutputType, modelLiteral, ...)                         \
   const std::string _modelName = (modelLiteral);                                     \
   ASSERT_TRUE(includeModel(_modelName)) << "Failed to include model " << _modelName; \
   auto output = runModel<OutputType>(#OutputType, _modelName, "", __VA_ARGS__);

#define ASSERT_INCLUDE_AND_RUN_NO_SESSION(OutputType, modelLiteral, ...)              \
   const std::string _modelName = (modelLiteral);                                     \
   ASSERT_TRUE(includeModel(_modelName)) << "Failed to include model " << _modelName; \
   auto output = runModel<OutputType>(#OutputType, _modelName, "NO_SESSION", __VA_ARGS__);

#define ASSERT_INCLUDE_AND_RUN_SESSION_ARGS(OutputType, modelLiteral, sessionArgs, ...) \
   const std::string _modelName = (modelLiteral);                                       \
   ASSERT_TRUE(includeModel(_modelName)) << "Failed to include model " << _modelName;   \
   auto output = runModel<OutputType>(#OutputType, _modelName, sessionArgs, __VA_ARGS__);
