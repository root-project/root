#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <TInterpreter.h>

constexpr float DEFAULT_TOLERANCE = 1e-3f;

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
   if constexpr (std::is_same_v<T, int>) {
      return std::to_string(ptr);
   }
   std::string out =
      TString::Format("reinterpret_cast<%s*>(0x%zx)", className.c_str(), reinterpret_cast<std::size_t>(&ptr)).Data();
   if (toRawPointer) {
      out += "->data()";
   }
   return out;
}

// Output type names without commas in the name, to be used in macro calls
using TupleFloatInt64_t = std::tuple<std::vector<float>, std::vector<int64_t>>;

template <typename OutputType_t, typename... Ts>
OutputType_t
runModel(std::string outputTypeName, std::string const &modelName, std::string sessionArgs, Ts const &...inputs)
{
   OutputType_t output;

   // The interpreter doesn't know about our aliases, to we convert them back
   if (outputTypeName == "TupleFloatInt64_t") {
      outputTypeName = "std::tuple<std::vector<float>, std::vector<int64_t>>";
   }

   // Helper: map C++ type -> string used in interpreter
   auto type_name = []<typename T>() {
      if constexpr (std::is_same_v<T, int>)
         return "int";
      else if constexpr (std::is_same_v<T, std::vector<float>>)
         return "std::vector<float>";
      else if constexpr (std::is_same_v<T, std::vector<int>>)
         return "std::vector<float>";
      else if constexpr (std::is_same_v<T, std::vector<int64_t>>)
         return "std::vector<int64_t>";
      else if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
         return "std::vector<uint8_t>";
      else
         static_assert(!sizeof(T), "Input type not supported");
   };

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
   bool first = true;
   (
      [&] {
         if (!first)
            cmd << ", ";
         first = false;
         cmd << toInterpreter(inputs, type_name.template operator()<Ts>(), true);
      }(),
      ...);

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
