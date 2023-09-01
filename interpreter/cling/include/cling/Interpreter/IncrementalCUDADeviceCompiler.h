//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_CUDA_DEVICE_JIT_H
#define CLING_INCREMENTAL_CUDA_DEVICE_JIT_H

#include "clang/Basic/CodeGenOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Triple.h"

#include <string>
#include <vector>

namespace cling {
  class InvocationOptions;
  class Transaction;
  class Interpreter;
} // namespace cling

namespace clang {
  class CompilerInstance;
  class HeaderSearchOptions;
  class LangOptions;
} // namespace clang

namespace llvm {
  class StringRef;
}

namespace cling {

  ///\brief The class is responsible for generating CUDA device code in
  /// cuda fatbinary form during the runtime. The PTX code is compiled by a
  /// second interpreter instance.
  ///
  class IncrementalCUDADeviceCompiler {

    ///\brief Contains the arguments for cling nvptx and flags for fatbinary
    /// generation
    struct CUDACompilerArgs {
      const std::string cppStdVersion;
      ///\brief contains information about host architecture
      const llvm::Triple hostTriple;
      const uint32_t smVersion;
      ///\brief see setCuArgs()
      const uint32_t fatbinFlags;
      ///\brief True, if the flag -v is set.
      const bool verbose;
      ///\brief True, if the flag -g is set.
      const bool debug;
      ///\brief A list Arguments, which will passed to the clang nvptx.
      const std::vector<std::string> additionalPtxOpt;

      CUDACompilerArgs(const std::string cppStdVersion,
                       const llvm::Triple hostTriple, const uint32_t smVersion,
                       const uint32_t fatbinFlags, const bool verbose,
                       const bool debug,
                       const std::vector<std::string> additionalPtxOpt)
          : cppStdVersion(cppStdVersion),
            hostTriple(hostTriple), smVersion(smVersion),
            fatbinFlags(fatbinFlags), verbose(verbose), debug(debug),
            additionalPtxOpt(additionalPtxOpt) {}
    };

    std::unique_ptr<CUDACompilerArgs> m_CuArgs;

    ///\brief Interpreter instance with target NVPTX which compiles the input to
    ///LLVM IR. Then the LLVM IR is compiled to PTX via an additional backend.
    std::unique_ptr<Interpreter> m_PTX_interp;

    ///\brief Is true if the second interpreter instance was created and the
    /// NVIDIA fatbin tool was found.
    bool m_Init = false;

    ///\brief Path to the folder, where all files will put in. Ordinary the tmp
    /// folder. Have to end with a separator. Can be empty.
    const std::string m_FilePath;
    ///\brief Path to the fatbin file, which will used by the CUDACodeGen.
    const std::string m_FatbinFilePath;

    ///\brief Contains the PTX code of the current input
    llvm::SmallString<1024> m_PTX_code;

    ///\brief Keep the ptx compiler args for reflection during runtime.
    std::vector<std::string> argv;

    ///\brief Add the include paths from the interpreter runtime to a argument
    /// list.
    ///
    ///\param [in,out] argv - The include commands will append to the argv
    /// vector.
    ///\param [in] headerSearchOptions - Contains information about all include
    /// paths.
    void addHeaderSearchPathFlags(
        std::vector<std::string>& argv,
        const std::shared_ptr<clang::HeaderSearchOptions> &headerSearchOptions);

    ///\brief Compiles a PTX file from the current input. The PTX code is
    /// written to cling.ptx.
    ///
    ///\returns True, if the new cling.ptx was compiled.
    bool generatePTX();

    ///\brief Wrap up the ptx_code in the NVIDIA fatbinary format. The fatbin
    /// code is written to m_FatbinFilePath.
    ///
    ///\returns True, if the fatbinary tool returns 0.
    bool generateFatbinary();

    ///\brief The function set the values of m_CuArgs.
    ///
    ///\param [in] langOpts - The LangOptions of the CompilerInstance.
    ///\param [in] invocationOptions - The invocationOptions of the interpreter.
    ///\param [in] debugInfo - The debugInfo of the CompilerInstance.
    ///\param [in] hostTriple - The llvm triple of the host system
    void setCuArgs(const clang::LangOptions& langOpts,
                   const cling::InvocationOptions& invocationOptions,
                   const clang::codegenoptions::DebugInfoKind debugInfo,
                   const llvm::Triple hostTriple);

  public:
    ///\brief Constructor for IncrementalCUDADeviceCompiler
    ///
    ///\param [in] filePath - All files will generated in the folder of the
    ///       filePath, except the fatbin file, if it have another path. Have
    ///       to end with a separator. Can be empty.
    ///\param [in] optLevel - The optimization level of the interpreter
    /// instance.
    ///       The value will be copied, because a change of it is not allowed.
    ///\param [in] invocationOptions - Contains values for the arguments of
    ///       the interpreter instance and the NVIDIA tool fatbinary.
    ///\param [in] CI - Will be used for m_CuArgs and the include path handling.
    IncrementalCUDADeviceCompiler(
        const std::string& filePath, const int optLevel,
        const cling::InvocationOptions& invocationOptions,
        const clang::CompilerInstance& CI);

    ///\brief Returns a reference to the PTX interpreter
    ///
    ///\return std::unique_ptr< cling::Interpreter >&
    ///
    Interpreter *getPTXInterpreter() { return m_PTX_interp.get(); }

    ///\brief Generate an new fatbin file with the path in
    /// CudaGpuBinaryFileNames.
    ///
    /// This interface helps to run everything that the device compiler can run.
    /// Note that this should be used when there is no idea of what kind of
    /// input is going to be processed. Otherwise if is known, for example
    /// only header files are going to be processed it is much faster to run the
    /// specific interface for doing that - in the particular case - declare().
    ///
    ///\param [in] input - The input directly from the UI. Attention, the string
    /// must not be wrapped or transformed.
    ///
    ///\returns true, if all stages of generating fatbin runs right and a new
    /// fatbin file is written.
    bool process(const std::string& input);

    ///\brief Generate an new fatbin file with the path in
    /// CudaGpuBinaryFileNames from input line, which doesn't contain
    /// statements.
    ///
    /// The interface circumvents the most of the extra work necessary to
    /// compile and run statements.
    ///
    /// @param[in] input - The input containing only declarations (aka
    ///                    Top Level Declarations)
    ///
    ///\returns true, if all stages of generating fatbin runs right and a new
    /// fatbin file is written.
    bool declare(const std::string& input);

    ///\brief Parses input line, which doesn't contain statements. No code
    /// generation is done.
    ///
    /// Same as declare without codegening. Useful when a library is loaded and
    /// the header files need to be imported.
    ///
    ///\param[in] input - The input containing the declarations.
    ///
    ///\returns true if parsing of the input was correct
    bool parse(const std::string& input) const;

    ///\brief Print some information of the IncrementalCUDADeviceCompiler to
    /// llvm::outs().
    void dump();
  };

} // namespace cling

#endif // CLING_INCREMENTAL_CUDA_DEVICE_JIT_H
