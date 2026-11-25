#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

#include "driver_runner.h"
#include "test_utils.h"

// Output type names without commas in the name, to be used in macro calls
using TupleFloatInt64_t = std::tuple<std::vector<float>, std::vector<int64_t>>;

/// The generated header of a model must exist
inline bool includeModel(std::string const &modelName)
{
   const std::string header = modelName + modelHeaderSuffix;
   if (std::ifstream(header))
      return true;
   std::cerr << "\n[includeModel] Header file " << header << " of model " << modelName << " not found\n";
   return false;
}

/// Argument of infer() for an input: vectors are passed as raw pointers
template <typename T>
std::string inputArgument(std::string const &name)
{
   return std::is_arithmetic_v<T> ? name : name + ".data()";
}

template <typename OutputType_t, typename... Ts>
OutputType_t
runModel(std::string outputTypeName, std::string const &modelName, std::string sessionArgs, Ts const &...inputs)
{
   // The driver doesn't know about our aliases, so we convert them back
   if (outputTypeName == "TupleFloatInt64_t") {
      outputTypeName = "std::tuple<std::vector<float>, std::vector<int64_t>>";
   }

   if (sessionArgs.empty()) {
      sessionArgs = R"(")" + modelName + modelDataSuffix + R"(")";
   }

   std::stringstream driver;
   driver << "#include \"" << modelName << modelHeaderSuffix << "\"\n"
          << "#include \"driver_support.hxx\"\n"
          << "#include <iostream>\n#include <limits>\n#include <vector>\n"
          << "int main() {\n   try {\n";

   // Declare all inputs and gather the arguments of infer(...)
   std::string arguments;
   std::size_t iInput = 0;
   auto addInput = [&](auto const &input) {
      using T = std::decay_t<decltype(input)>;
      const std::string name = "input" + std::to_string(iInput++);
      declareInput(driver, name, input);
      arguments += (arguments.empty() ? "" : ", ") + inputArgument<T>(name);
   };
   (addInput(inputs), ...);

   if (sessionArgs != "NO_SESSION") {
      driver << "   SOFIE_" << modelName << "::Session s(" << sessionArgs << ");\n"
             << "   " << outputTypeName << " output = s.infer(" << arguments << ");\n";
   } else {
      driver << "   " << outputTypeName << " output = SOFIE_" << modelName << "::infer(" << arguments
             << ");\n";
   }
   driver << "   sofie_test::write(std::cout, output);\n"
          << "   } catch (std::exception const &e) {\n"
          << "      std::cerr << \"exception: \" << e.what() << std::endl;\n"
          << "      return 1;\n"
          << "   }\n   return 0;\n}\n";

   DriverResult result = compileAndRunDriver(driver.str());
   if (!checkDriver(result, "model " + modelName))
      return OutputType_t{};

   std::istringstream in(result.output);
   return sofie_test::Reader{in}.read<OutputType_t>();
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
