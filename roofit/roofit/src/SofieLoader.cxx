#include "SofieLoader.h"

#include <RConfigure.h>
#include <TString.h>
#include <TSystem.h>

#include <memory>
#include <stdexcept>

namespace {

// The functions of the C interface of SOFIE (see SOFIE/SOFIE.h in the sources of
// SOFIE), which are looked up at runtime
using GenerateFunc_t = std::size_t (*)(const char *onnx, std::size_t size, const char *modelName, unsigned int flags,
                                       char **code, char **error);
using FreeFunc_t = void (*)(char *str);

// The flags of SOFIE_GenerateCodeFromONNX
constexpr unsigned int kSofieUseBlas = 1u << 0;
#ifdef R__HAS_VDT
constexpr unsigned int kSofieUseVdt = 1u << 1;
#endif

constexpr const char *kSofieLibrary = "libSOFIE";

struct SofieFunctions {
   GenerateFunc_t fGenerate = nullptr;
   FreeFunc_t fFree = nullptr;
   /// Why the functions are not available
   std::string fError;
};

SofieFunctions LoadSofie()
{
   SofieFunctions functions;
   TString library{kSofieLibrary};
   // The library is looked for next to the ROOT libraries, where the SOFIE that
   // is built together with ROOT is installed, and in the library path. ROOT
   // doesn't know where an external SOFIE is installed, since there is no
   // FindSOFIE.cmake module or SOFIEConfig.cmake package file for it yet: its
   // library has to be in the library path (e.g. in LD_LIBRARY_PATH).
   // Look for the library first, so that nothing is printed if it is not found
   if (!gSystem->FindDynamicLibrary(library, /*quiet=*/true) || gSystem->Load(library) < 0) {
      functions.fError = std::string{"the SOFIE library ("} + kSofieLibrary +
                         "), which is needed to evaluate ONNX models, can't be loaded. It is built "
                         "together with ROOT if roofit or dataframe are enabled. An external SOFIE library "
                         "has to be in the library path";
      return functions;
   }
   functions.fGenerate =
      reinterpret_cast<GenerateFunc_t>(gSystem->DynFindSymbol(library, "SOFIE_GenerateCodeFromONNX"));
   functions.fFree = reinterpret_cast<FreeFunc_t>(gSystem->DynFindSymbol(library, "SOFIE_Free"));
   if (!functions.fGenerate || !functions.fFree)
      functions.fError = "the library " + std::string{library.Data()} + " doesn't provide the interface of SOFIE";
   return functions;
}

/// The functions of SOFIE, which are looked up only once
SofieFunctions const &GetSofie()
{
   static const SofieFunctions functions = LoadSofie();
   return functions;
}

/// The flags to use the BLAS library and vdt in the generated code, if they are available
unsigned int GetFlags()
{
   unsigned int flags = 0;
   // The generated code calls the routines of the BLAS library if it is loaded
   // in the process, and defines them otherwise
   if (gSystem->DynFindSymbol("*", "sgemm_") && gSystem->DynFindSymbol("*", "saxpy_"))
      flags |= kSofieUseBlas;
#ifdef R__HAS_VDT
   // The generated code uses vdt only if its headers are found when it is compiled
   flags |= kSofieUseVdt;
#endif
   return flags;
}

} // namespace

bool RooFit::Detail::IsSofieAvailable()
{
   return GetSofie().fError.empty();
}

std::string RooFit::Detail::GenerateCodeFromONNX(std::string_view onnx, std::string const &modelName)
{
   SofieFunctions const &sofie = GetSofie();
   if (!sofie.fError.empty())
      throw std::runtime_error("RooFit: " + sofie.fError);

   char *code = nullptr;
   char *error = nullptr;
   const std::size_t length = sofie.fGenerate(onnx.data(), onnx.size(), modelName.c_str(), GetFlags(), &code, &error);
   // The strings are allocated by SOFIE, which releases them
   std::unique_ptr<char, FreeFunc_t> codeOwner{code, sofie.fFree};
   std::unique_ptr<char, FreeFunc_t> errorOwner{error, sofie.fFree};
   if (length == 0)
      throw std::runtime_error("RooFit: SOFIE failed to generate the code of the ONNX model " + modelName + ": " +
                               (error ? error : "unknown error"));
   return {code, length};
}
