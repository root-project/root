// Compile and run small programs (drivers) around generated models. This is
// how the tests run generated code: with the C++ compiler used to build the
// tests, in a separate process.
#pragma once

#include "driver_support.hxx"
#include "test_config.h"

#include <gtest/gtest.h>

#include <sys/wait.h>

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

/// Result of compiling and running a driver
struct DriverResult {
   bool compiled = false;
   int exitCode = -1;
   std::string compileLog; ///< output of the compiler
   std::string output;     ///< what the driver wrote to the standard output
   std::string errors;     ///< what the driver wrote to the standard error
};

inline std::string readTextFile(std::string const &path)
{
   std::ifstream in(path);
   return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline std::string shellQuote(std::string const &s)
{
   std::string out = "'";
   for (char c : s) {
      if (c == '\'')
         out += "'\\''";
      else
         out += c;
   }
   return out + "'";
}

/// Compile a driver (a complete program) with the C++ compiler used to build the
/// tests, and run it from the current directory, where the generated models are.
/// The driver can include the generated headers and driver_support.hxx. Extra
/// compiler flags can be passed, e.g. to use a compiler plugin.
inline DriverResult compileAndRunDriver(std::string const &source, std::string const &extraCompilerFlags = "")
{
   namespace fs = std::filesystem;
   static std::atomic<int> counter{0};

   std::string tag = "driver";
   if (auto const *info = ::testing::UnitTest::GetInstance()->current_test_info())
      tag = std::string(info->test_suite_name()) + "_" + info->name();
   for (auto &c : tag)
      if (!std::isalnum(static_cast<unsigned char>(c)))
         c = '_';

   const fs::path cwd = fs::current_path();
   const fs::path dir = cwd / "drivers" / (tag + "_" + std::to_string(counter++));
   fs::create_directories(dir);

   {
      std::ofstream out(dir / "main.cpp");
      out << source;
   }

   DriverResult result;
   const std::string compile = shellQuote(SOFIE_TEST_CXX) + " " + SOFIE_TEST_CXX_FLAGS + " -O0 -w " +
                               extraCompilerFlags + " -I" + shellQuote(cwd.string()) + " -I" +
                               shellQuote(SOFIE_TEST_SOURCE_DIR) + " " + shellQuote((dir / "main.cpp").string()) +
                               " -o " + shellQuote((dir / "main").string()) + " > " +
                               shellQuote((dir / "compile.log").string()) + " 2>&1";
   result.compiled = std::system(compile.c_str()) == 0;
   result.compileLog = readTextFile((dir / "compile.log").string());
   if (!result.compiled)
      return result;

   const std::string run = shellQuote((dir / "main").string()) + " > " + shellQuote((dir / "out.txt").string()) +
                           " 2> " + shellQuote((dir / "err.txt").string());
   const int status = std::system(run.c_str());
   result.exitCode = (status != -1 && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
   result.output = readTextFile((dir / "out.txt").string());
   result.errors = readTextFile((dir / "err.txt").string());
   return result;
}

/// Report a driver that did not compile or run successfully. Returns true if it did.
inline bool checkDriver(DriverResult const &result, std::string const &what)
{
   if (!result.compiled) {
      ADD_FAILURE() << "Failed to compile the driver for " << what << ":\n" << result.compileLog;
      return false;
   }
   if (result.exitCode != 0) {
      ADD_FAILURE() << "The driver for " << what << " failed with exit code " << result.exitCode << ":\n"
                    << result.errors;
      return false;
   }
   return true;
}

template <typename T>
constexpr const char *typeName()
{
   if constexpr (std::is_same_v<T, int>)
      return "int";
   else if constexpr (std::is_same_v<T, std::size_t>)
      return "std::size_t";
   else if constexpr (std::is_same_v<T, float>)
      return "float";
   else if constexpr (std::is_same_v<T, double>)
      return "double";
   else if constexpr (std::is_same_v<T, int64_t>)
      return "int64_t";
   else if constexpr (std::is_same_v<T, uint8_t>)
      return "uint8_t";
   else
      static_assert(!sizeof(T), "Input type not supported");
}

/// C++ literal of an arithmetic value
template <typename T>
std::string literal(T value)
{
   std::stringstream out;
   if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(value))
         out << "std::numeric_limits<" << typeName<T>() << ">::quiet_NaN()";
      else if (std::isinf(value))
         out << (value < 0 ? "-" : "") << "std::numeric_limits<" << typeName<T>() << ">::infinity()";
      else
         out << std::hexfloat << value << (std::is_same_v<T, float> ? "f" : "");
   } else {
      out << static_cast<long long>(value);
   }
   return out.str();
}

/// Declare the variable of the given name in the driver, with the value of an input
template <typename T>
void declareInput(std::stringstream &driver, std::string const &name, T const &input)
{
   if constexpr (std::is_arithmetic_v<T>) {
      driver << "   " << typeName<T>() << " " << name << " = " << literal(input) << ";\n";
   } else {
      using Element = typename T::value_type;
      driver << "   std::vector<" << typeName<Element>() << "> " << name << " = {";
      for (std::size_t i = 0; i < input.size(); ++i)
         driver << (i ? ", " : "") << literal(input[i]);
      driver << "};\n";
   }
}
