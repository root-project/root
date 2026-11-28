/// \file SOFIE.cxx
/// Implementation of the C interface of the SOFIE library.

#include "SOFIE/SOFIE.h"

#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelParser_ONNX.hxx"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

/// Copy a string to a buffer allocated with malloc, to be released with SOFIE_Free.
char *CopyString(const std::string &str)
{
   auto *out = static_cast<char *>(std::malloc(str.size() + 1));
   if (out)
      std::memcpy(out, str.c_str(), str.size() + 1);
   return out;
}

std::string GenerateCode(const char *onnx, size_t size, const char *modelName, unsigned int flags)
{
   if (!onnx || size == 0)
      throw std::invalid_argument("SOFIE: the content of the ONNX file is empty");
   if (!modelName || modelName[0] == '\0')
      throw std::invalid_argument("SOFIE: the name of the model is empty");
   constexpr unsigned int knownFlags = SOFIE_USE_BLAS | SOFIE_USE_VDT;
   if (flags & ~knownFlags)
      throw std::invalid_argument("SOFIE: unknown flags " + std::to_string(flags & ~knownFlags));

   std::istringstream stream{std::string{onnx, size}};
   SOFIE::RModel model = SOFIE::RModelParser_ONNX{}.Parse(stream, modelName);
   model.SetOptimizationLevel(SOFIE::OptimizationLevel::kBasic);
   model.SetUseBlasLibrary(flags & SOFIE_USE_BLAS);
   model.SetUseVDT(flags & SOFIE_USE_VDT);
   model.Generate(SOFIE::Options::kNoWeightFile);

   std::ostringstream code;
   model.PrintGenerated(code);
   return code.str();
}

} // namespace

// The whole function is a function try-block, so that no exception can cross the C interface
size_t SOFIE_GenerateCodeFromONNX(const char *onnx, size_t size, const char *modelName, unsigned int flags, char **code,
                                  char **error)
try {
   if (error)
      *error = nullptr;
   if (!code)
      throw std::invalid_argument("SOFIE: the output argument for the code is NULL");
   *code = nullptr;

   const std::string generatedCode = GenerateCode(onnx, size, modelName, flags);
   *code = CopyString(generatedCode);
   if (!*code)
      throw std::bad_alloc{};
   return generatedCode.size();
} catch (const std::exception &e) {
   if (error)
      *error = CopyString(e.what());
   return 0;
} catch (...) {
   if (error)
      *error = CopyString("SOFIE: unknown error");
   return 0;
}

void SOFIE_Free(char *str)
{
   std::free(str);
}
