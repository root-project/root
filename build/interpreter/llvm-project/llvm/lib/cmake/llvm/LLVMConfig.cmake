# This file provides information and services to the final user.


# LLVM_BUILD_* values available only from LLVM build tree.
set(LLVM_BUILD_BINARY_DIR "/home/runner/work/root/root/build/interpreter/llvm-project/llvm")
set(LLVM_BUILD_LIBRARY_DIR "/home/runner/work/root/root/build/interpreter/llvm-project/llvm/./lib")
set(LLVM_BUILD_MAIN_INCLUDE_DIR "/home/runner/work/root/root/interpreter/llvm-project/llvm/include")
set(LLVM_BUILD_MAIN_SRC_DIR "/home/runner/work/root/root/interpreter/llvm-project/llvm")


# For finding self-installed Find*.cmake packages.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

set(LLVM_VERSION_MAJOR 18)
set(LLVM_VERSION_MINOR 1)
set(LLVM_VERSION_PATCH 8)
set(LLVM_VERSION_SUFFIX )
set(LLVM_PACKAGE_VERSION 18.1.8)
set(LLVM_PACKAGE_BUGREPORT https://github.com/llvm/llvm-project/issues/)

set(LLVM_BUILD_TYPE Release)

set(CMAKE_MSVC_RUNTIME_LIBRARY )

set(LLVM_USE_SPLIT_DWARF OFF)

set(LLVM_COMMON_DEPENDS )

set(LLVM_AVAILABLE_LIBS LLVMDemangle;LLVMSupport;LLVMTableGen;LLVMTableGenGlobalISel;LLVMTableGenCommon;LLVMCore;LLVMFuzzerCLI;LLVMFuzzMutate;LLVMFileCheck;LLVMInterfaceStub;LLVMIRPrinter;LLVMIRReader;LLVMCodeGenTypes;LLVMCodeGen;LLVMSelectionDAG;LLVMAsmPrinter;LLVMMIRParser;LLVMGlobalISel;LLVMBinaryFormat;LLVMBitReader;LLVMBitWriter;LLVMBitstreamReader;LLVMDWARFLinker;LLVMDWARFLinkerClassic;LLVMDWARFLinkerParallel;LLVMExtensions;LLVMFrontendDriver;LLVMFrontendHLSL;LLVMFrontendOpenACC;LLVMFrontendOpenMP;LLVMFrontendOffloading;LLVMTransformUtils;LLVMInstrumentation;LLVMAggressiveInstCombine;LLVMInstCombine;LLVMScalarOpts;LLVMipo;LLVMVectorize;LLVMObjCARCOpts;LLVMCoroutines;LLVMCFGuard;LLVMHipStdPar;LLVMLinker;LLVMAnalysis;LLVMLTO;LLVMMC;LLVMMCParser;LLVMMCDisassembler;LLVMMCA;LLVMObjCopy;LLVMObject;LLVMObjectYAML;LLVMOption;LLVMRemarks;LLVMDebuginfod;LLVMDebugInfoDWARF;LLVMDebugInfoGSYM;LLVMDebugInfoLogicalView;LLVMDebugInfoMSF;LLVMDebugInfoCodeView;LLVMDebugInfoPDB;LLVMSymbolize;LLVMDebugInfoBTF;LLVMDWP;LLVMExecutionEngine;LLVMInterpreter;LLVMJITLink;LLVMMCJIT;LLVMOrcJIT;LLVMOrcDebugging;LLVMOrcShared;LLVMOrcTargetProcess;LLVMRuntimeDyld;LLVMTarget;LLVMX86CodeGen;LLVMX86AsmParser;LLVMX86Disassembler;LLVMX86TargetMCA;LLVMX86Desc;LLVMX86Info;LLVMNVPTXCodeGen;LLVMNVPTXDesc;LLVMNVPTXInfo;LLVMAsmParser;LLVMLineEditor;LLVMProfileData;LLVMCoverage;LLVMPasses;LLVMTargetParser;LLVMTextAPI;LLVMTextAPIBinaryReader;LLVMDlltoolDriver;LLVMLibDriver;LLVMXRay;LLVMWindowsDriver;LLVMWindowsManifest;LTO;LLVMCFIVerify;LLVMDiff;LLVMExegesisX86;LLVMExegesis;Remarks)



set(LLVM_DYLIB_COMPONENTS all)

set(LLVM_ALL_TARGETS AArch64;AMDGPU;ARM;AVR;BPF;Hexagon;Lanai;LoongArch;Mips;MSP430;NVPTX;PowerPC;RISCV;Sparc;SystemZ;VE;WebAssembly;X86;XCore)

set(LLVM_TARGETS_TO_BUILD X86;NVPTX)

set(LLVM_TARGETS_WITH_JIT X86;PowerPC;AArch64;ARM;Mips;SystemZ)


set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDemangle )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMSupport )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTableGen )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTableGenGlobalISel )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTableGenCommon )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMCore )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFuzzerCLI )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFuzzMutate )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFileCheck )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMInterfaceStub )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMIRPrinter )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMIRReader )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMCodeGenTypes )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMCodeGen )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMSelectionDAG )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMAsmPrinter )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMMIRParser )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMGlobalISel )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMBinaryFormat )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMBitReader )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMBitWriter )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMBitstreamReader )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDWARFLinker )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDWARFLinkerClassic )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDWARFLinkerParallel )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMExtensions )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFrontendDriver )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFrontendHLSL )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFrontendOpenACC )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFrontendOpenMP )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMFrontendOffloading )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTransformUtils )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMInstrumentation )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMAggressiveInstCombine )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMInstCombine )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMScalarOpts )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMipo )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMVectorize )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMObjCARCOpts )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMCoroutines )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMCFGuard )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMHipStdPar )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMLinker )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMAnalysis )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMLTO )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMMC )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMMCParser )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMMCDisassembler )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMMCA )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMObjCopy )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMObject )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMObjectYAML )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMOption )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMRemarks )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebuginfod )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebugInfoDWARF )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebugInfoGSYM )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebugInfoLogicalView )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebugInfoMSF )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebugInfoCodeView )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebugInfoPDB )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMSymbolize )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDebugInfoBTF )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDWP )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMExecutionEngine )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMInterpreter )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMJITLink )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMMCJIT )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMOrcJIT )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMOrcDebugging )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMOrcShared )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMOrcTargetProcess )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMRuntimeDyld )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTarget )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMX86CodeGen )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMX86AsmParser )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMX86Disassembler )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMX86TargetMCA )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMX86Desc )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMX86Info )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMNVPTXCodeGen )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMNVPTXDesc )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMNVPTXInfo )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMAsmParser )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMLineEditor )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMProfileData )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMCoverage )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMPasses )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTargetParser )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTextAPI )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMTextAPIBinaryReader )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDlltoolDriver )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMLibDriver )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMXRay )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMWindowsDriver )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMWindowsManifest )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LTO )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMCFIVerify )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMDiff )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMExegesisX86 )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_LLVMExegesis )
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_Remarks )

set(LLVM_TARGET_TRIPLE "x86_64-unknown-linux-gnu")

set(LLVM_HOST_TRIPLE "x86_64-unknown-linux-gnu")

set(LLVM_ABI_BREAKING_CHECKS WITH_ASSERTS)

set(LLVM_ENABLE_WARNINGS OFF)

set(LLVM_ENABLE_EXPENSIVE_CHECKS OFF)

set(LLVM_ENABLE_ASSERTIONS FALSE)

set(LLVM_ENABLE_EH OFF)

set(LLVM_ENABLE_FFI OFF)
if(LLVM_ENABLE_FFI)
  find_package(FFI)
endif()

set(LLVM_ENABLE_RTTI ON)

set(LLVM_ENABLE_LIBEDIT FALSE)
if(LLVM_ENABLE_LIBEDIT)
  find_package(LibEdit)
endif()

set(LLVM_ENABLE_TERMINFO TRUE)
if(LLVM_ENABLE_TERMINFO)
  find_package(Terminfo)
endif()

set(LLVM_ENABLE_THREADS OFF)

set(LLVM_ENABLE_UNWIND_TABLES ON)

set(LLVM_ENABLE_ZLIB 1)
if(LLVM_ENABLE_ZLIB)
  set(ZLIB_ROOT )
  find_package(ZLIB)
endif()

set(LLVM_ENABLE_ZSTD TRUE)
if(LLVM_ENABLE_ZSTD)
  find_package(zstd)
endif()

set(LLVM_ENABLE_LIBXML2 1)
if(LLVM_ENABLE_LIBXML2)
  find_package(LibXml2)
endif()

set(LLVM_ENABLE_CURL OFF)
if(LLVM_ENABLE_CURL)
  find_package(CURL)
endif()

set(LLVM_ENABLE_HTTPLIB OFF)
if(LLVM_ENABLE_HTTPLIB)
  find_package(httplib)
endif()

set(LLVM_WITH_Z3 )

set(LLVM_ENABLE_DIA_SDK 0)

set(LLVM_NATIVE_ARCH X86)

set(LLVM_ENABLE_PIC ON)

set(LLVM_BUILD_32_BITS OFF)

if (NOT "" STREQUAL "")
  set(LLVM_PTHREAD_LIB "")
endif()

set(LLVM_ENABLE_PLUGINS ON)
set(LLVM_EXPORT_SYMBOLS_FOR_PLUGINS OFF)
set(LLVM_PLUGIN_EXT .so)

set(LLVM_ON_UNIX 1)

set(LLVM_LIBDIR_SUFFIX )

set(LLVM_INCLUDE_DIRS "/home/runner/work/root/root/interpreter/llvm-project/llvm/include;/home/runner/work/root/root/build/interpreter/llvm-project/llvm/include")
set(LLVM_LIBRARY_DIRS "/home/runner/work/root/root/build/interpreter/llvm-project/llvm/./lib")

set(LLVM_APPEND_VC_REV "OFF")

# These variables are duplicated for install tree but they have different
# values for build tree.  LLVM_INCLUDE_DIRS contains both source
# and generated include directories while the following variables have
# them split.

# These are the "main" dirs
set(LLVM_MAIN_INCLUDE_DIR "/home/runner/work/root/root/interpreter/llvm-project/llvm/include")
set(LLVM_LIBRARY_DIR "/home/runner/work/root/root/build/interpreter/llvm-project/llvm/./lib")

# This is a secondary one for generated files
set(LLVM_INCLUDE_DIR "/home/runner/work/root/root/build/interpreter/llvm-project/llvm/include")

set(LLVM_DEFINITIONS "")
set(LLVM_BINARY_DIR "/home/runner/work/root/root/build/interpreter/llvm-project/llvm")
set(LLVM_CMAKE_DIR "/home/runner/work/root/root/interpreter/llvm-project/llvm/cmake/modules")
set(LLVM_TOOLS_BINARY_DIR "/home/runner/work/root/root/build/interpreter/llvm-project/llvm/./bin")
set(LLVM_HAVE_OPT_VIEWER_MODULES 1)
set(LLVM_CONFIGURATION_TYPES )
set(LLVM_ENABLE_SHARED_LIBS FALSE)

set(LLVM_DEFAULT_EXTERNAL_LIT "/home/runner/work/root/root/build/interpreter/llvm-project/llvm/./bin/llvm-lit")
set(LLVM_LIT_ARGS "-sv")

if(NOT TARGET LLVMSupport)
  include("/home/runner/work/root/root/build/interpreter/llvm-project/llvm/./lib/cmake/llvm/LLVMExports.cmake")
  include("/home/runner/work/root/root/build/interpreter/llvm-project/llvm/./lib/cmake/llvm/LLVMBuildTreeOnlyTargets.cmake")
endif()

# By creating the following targets here, subprojects that depend on
# LLVM's tablegen-generated headers can always depend on this target
# whether building in-tree with LLVM or not.
if(NOT TARGET intrinsics_gen)
  add_custom_target(intrinsics_gen)
endif()
if(NOT TARGET omp_gen)
  add_custom_target(omp_gen)
endif()
if(NOT TARGET acc_gen)
  add_custom_target(acc_gen)
endif()
if(NOT TARGET RISCVTargetParserTableGen)
  add_custom_target(RISCVTargetParserTableGen)
endif()

set_property(GLOBAL PROPERTY LLVM_TARGETS_CONFIGURED On)
include(${LLVM_CMAKE_DIR}/LLVM-Config.cmake)
