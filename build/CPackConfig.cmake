# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BINARY_DEB "OFF")
set(CPACK_BINARY_FREEBSD "OFF")
set(CPACK_BINARY_IFW "OFF")
set(CPACK_BINARY_NSIS "OFF")
set(CPACK_BINARY_RPM "OFF")
set(CPACK_BINARY_STGZ "ON")
set(CPACK_BINARY_TBZ2 "OFF")
set(CPACK_BINARY_TGZ "ON")
set(CPACK_BINARY_TXZ "OFF")
set(CPACK_BINARY_TZ "ON")
set(CPACK_BUILD_SOURCE_DIRS "/home/runner/work/root/root;/home/runner/work/root/root/build")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENTS_ALL "LLVMAggressiveInstCombine;LLVMAnalysis;LLVMAsmParser;LLVMAsmPrinter;LLVMBinaryFormat;LLVMBitReader;LLVMBitWriter;LLVMBitstreamReader;LLVMCFGuard;LLVMCFIVerify;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMCoroutines;LLVMCoverage;LLVMDWARFLinker;LLVMDWARFLinkerClassic;LLVMDWARFLinkerParallel;LLVMDWP;LLVMDebugInfoBTF;LLVMDebugInfoCodeView;LLVMDebugInfoDWARF;LLVMDebugInfoGSYM;LLVMDebugInfoLogicalView;LLVMDebugInfoMSF;LLVMDebugInfoPDB;LLVMDebuginfod;LLVMDemangle;LLVMDiff;LLVMDlltoolDriver;LLVMExecutionEngine;LLVMExegesis;LLVMExegesisX86;LLVMExtensions;LLVMFileCheck;LLVMFrontendDriver;LLVMFrontendHLSL;LLVMFrontendOffloading;LLVMFrontendOpenACC;LLVMFrontendOpenMP;LLVMFuzzMutate;LLVMFuzzerCLI;LLVMGlobalISel;LLVMHipStdPar;LLVMIRPrinter;LLVMIRReader;LLVMInstCombine;LLVMInstrumentation;LLVMInterfaceStub;LLVMInterpreter;LLVMJITLink;LLVMLTO;LLVMLibDriver;LLVMLineEditor;LLVMLinker;LLVMMC;LLVMMCA;LLVMMCDisassembler;LLVMMCJIT;LLVMMCParser;LLVMMIRParser;LLVMNVPTXCodeGen;LLVMNVPTXDesc;LLVMNVPTXInfo;LLVMObjCARCOpts;LLVMObjCopy;LLVMObject;LLVMObjectYAML;LLVMOption;LLVMOrcDebugging;LLVMOrcJIT;LLVMOrcShared;LLVMOrcTargetProcess;LLVMPasses;LLVMProfileData;LLVMRemarks;LLVMRuntimeDyld;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMSymbolize;LLVMTableGen;LLVMTableGenCommon;LLVMTableGenGlobalISel;LLVMTarget;LLVMTargetParser;LLVMTextAPI;LLVMTextAPIBinaryReader;LLVMTransformUtils;LLVMVectorize;LLVMWindowsDriver;LLVMWindowsManifest;LLVMX86AsmParser;LLVMX86CodeGen;LLVMX86Desc;LLVMX86Disassembler;LLVMX86Info;LLVMX86TargetMCA;LLVMXRay;LLVMipo;LTO;Remarks;Unspecified;aarch64-resource-headers;applications;arm-common-resource-headers;arm-resource-headers;bash-autocomplete;clang;clang-cmake-exports;clang-cpp;clang-headers;clang-resource-headers;clang-tblgen;clangAPINotes;clangAST;clangASTMatchers;clangAnalysis;clangAnalysisFlowSensitive;clangAnalysisFlowSensitiveModels;clangBasic;clangCodeGen;clangCrossTU;clangDependencyScanning;clangDirectoryWatcher;clangDriver;clangDynamicASTMatchers;clangEdit;clangExtractAPI;clangFormat;clangFrontend;clangFrontendTool;clangIndex;clangIndexSerialization;clangInterpreter;clangLex;clangParse;clangRewrite;clangRewriteFrontend;clangSema;clangSerialization;clangStaticAnalyzerCheckers;clangStaticAnalyzerCore;clangStaticAnalyzerFrontend;clangSupport;clangTooling;clangToolingASTDiff;clangToolingCore;clangToolingInclusions;clangToolingInclusionsStdlib;clangToolingRefactoring;clangToolingSyntax;clangTransformer;cling-cmake-exports;clingInterpreter;clingMetaProcessor;clingUtils;cmake-exports;core-resource-headers;cuda-resource-headers;headers;hexagon-resource-headers;hip-resource-headers;hlsl-resource-headers;hmaptool;libraries;llvm-headers;llvm-tblgen;loongarch-resource-headers;mips-resource-headers;opencl-resource-headers;openmp-resource-headers;opt-viewer;ppc-htm-resource-headers;ppc-resource-headers;riscv-resource-headers;systemz-resource-headers;tests;utility-resource-headers;ve-resource-headers;webassembly-resource-headers;windows-resource-headers;x86-resource-headers")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/usr/local/share/cmake-3.31/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "ROOT built using CMake")
set(CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE "ON")
set(CPACK_GENERATOR "TGZ")
set(CPACK_INNOSETUP_ARCHITECTURE "x64")
set(CPACK_INSTALL_CMAKE_PROJECTS "/home/runner/work/root/root/build;ROOT;ALL;/")
set(CPACK_INSTALL_PREFIX "/usr/local")
set(CPACK_MODULE_PATH "/home/runner/work/root/root/cmake/modules")
set(CPACK_NSIS_DISPLAY_NAME "root_v6.37.01")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "root_v6.37.01")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OBJCOPY_EXECUTABLE "/usr/bin/objcopy")
set(CPACK_OBJDUMP_EXECUTABLE "/usr/bin/objdump")
set(CPACK_OUTPUT_CONFIG_FILE "/home/runner/work/root/root/build/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION "ROOT project")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/home/runner/work/root/root/build/README.md")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ROOT project")
set(CPACK_PACKAGE_EXECUTABLES "root;ROOT")
set(CPACK_PACKAGE_FILE_NAME "root_v6.37.01.Linux-ubuntu24.04-x86_64-gcc13.3")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "root_v6.37.01")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "root_v6.37.01")
set(CPACK_PACKAGE_NAME "ROOT")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "ROOT project")
set(CPACK_PACKAGE_VERSION "6.37.01")
set(CPACK_PACKAGE_VERSION_MAJOR "6")
set(CPACK_PACKAGE_VERSION_MINOR "37")
set(CPACK_PACKAGE_VERSION_PATCH "01")
set(CPACK_PROJECT_CONFIG_FILE "/home/runner/work/root/root/build/CMakeCPackOptions.cmake")
set(CPACK_READELF_EXECUTABLE "/usr/bin/readelf")
set(CPACK_RESOURCE_FILE_LICENSE "/home/runner/work/root/root/build/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "/home/runner/work/root/root/build/README.md")
set(CPACK_RESOURCE_FILE_WELCOME "/usr/local/share/cmake-3.31/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TGZ;TBZ2")
set(CPACK_SOURCE_IGNORE_FILES "/home/runner/work/root/root/build;/home/runner/work/root/root/tests;~$;/CVS/;/.svn/;/\\\\.svn/;/.git/;/\\\\.git/;\\\\.swp$;\\\\.swp$;\\.swp;\\\\.#;/#")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/runner/work/root/root/build/CPackSourceConfig.cmake")
set(CPACK_SOURCE_RPM "OFF")
set(CPACK_SOURCE_STRIP_FILES "")
set(CPACK_SOURCE_TBZ2 "ON")
set(CPACK_SOURCE_TGZ "ON")
set(CPACK_SOURCE_TXZ "ON")
set(CPACK_SOURCE_TZ "ON")
set(CPACK_SOURCE_ZIP "OFF")
set(CPACK_SYSTEM_NAME "Linux")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Linux")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/home/runner/work/root/root/build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()

# Configuration for installation type "full"
list(APPEND CPACK_ALL_INSTALL_TYPES full)
set(CPACK_INSTALL_TYPE_FULL_DISPLAY_NAME "Full Installation")

# Configuration for installation type "minimal"
list(APPEND CPACK_ALL_INSTALL_TYPES minimal)
set(CPACK_INSTALL_TYPE_MINIMAL_DISPLAY_NAME "Minimal Installation")

# Configuration for installation type "developer"
list(APPEND CPACK_ALL_INSTALL_TYPES developer)
set(CPACK_INSTALL_TYPE_DEVELOPER_DISPLAY_NAME "Developer Installation")

# Configuration for component "applications"

SET(CPACK_COMPONENTS_ALL LLVMAggressiveInstCombine LLVMAnalysis LLVMAsmParser LLVMAsmPrinter LLVMBinaryFormat LLVMBitReader LLVMBitWriter LLVMBitstreamReader LLVMCFGuard LLVMCFIVerify LLVMCodeGen LLVMCodeGenTypes LLVMCore LLVMCoroutines LLVMCoverage LLVMDWARFLinker LLVMDWARFLinkerClassic LLVMDWARFLinkerParallel LLVMDWP LLVMDebugInfoBTF LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoGSYM LLVMDebugInfoLogicalView LLVMDebugInfoMSF LLVMDebugInfoPDB LLVMDebuginfod LLVMDemangle LLVMDiff LLVMDlltoolDriver LLVMExecutionEngine LLVMExegesis LLVMExegesisX86 LLVMExtensions LLVMFileCheck LLVMFrontendDriver LLVMFrontendHLSL LLVMFrontendOffloading LLVMFrontendOpenACC LLVMFrontendOpenMP LLVMFuzzMutate LLVMFuzzerCLI LLVMGlobalISel LLVMHipStdPar LLVMIRPrinter LLVMIRReader LLVMInstCombine LLVMInstrumentation LLVMInterfaceStub LLVMInterpreter LLVMJITLink LLVMLTO LLVMLibDriver LLVMLineEditor LLVMLinker LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCJIT LLVMMCParser LLVMMIRParser LLVMNVPTXCodeGen LLVMNVPTXDesc LLVMNVPTXInfo LLVMObjCARCOpts LLVMObjCopy LLVMObject LLVMObjectYAML LLVMOption LLVMOrcDebugging LLVMOrcJIT LLVMOrcShared LLVMOrcTargetProcess LLVMPasses LLVMProfileData LLVMRemarks LLVMRuntimeDyld LLVMScalarOpts LLVMSelectionDAG LLVMSupport LLVMSymbolize LLVMTableGen LLVMTableGenCommon LLVMTableGenGlobalISel LLVMTarget LLVMTargetParser LLVMTextAPI LLVMTextAPIBinaryReader LLVMTransformUtils LLVMVectorize LLVMWindowsDriver LLVMWindowsManifest LLVMX86AsmParser LLVMX86CodeGen LLVMX86Desc LLVMX86Disassembler LLVMX86Info LLVMX86TargetMCA LLVMXRay LLVMipo LTO Remarks Unspecified aarch64-resource-headers applications arm-common-resource-headers arm-resource-headers bash-autocomplete clang clang-cmake-exports clang-cpp clang-headers clang-resource-headers clang-tblgen clangAPINotes clangAST clangASTMatchers clangAnalysis clangAnalysisFlowSensitive clangAnalysisFlowSensitiveModels clangBasic clangCodeGen clangCrossTU clangDependencyScanning clangDirectoryWatcher clangDriver clangDynamicASTMatchers clangEdit clangExtractAPI clangFormat clangFrontend clangFrontendTool clangIndex clangIndexSerialization clangInterpreter clangLex clangParse clangRewrite clangRewriteFrontend clangSema clangSerialization clangStaticAnalyzerCheckers clangStaticAnalyzerCore clangStaticAnalyzerFrontend clangSupport clangTooling clangToolingASTDiff clangToolingCore clangToolingInclusions clangToolingInclusionsStdlib clangToolingRefactoring clangToolingSyntax clangTransformer cling-cmake-exports clingInterpreter clingMetaProcessor clingUtils cmake-exports core-resource-headers cuda-resource-headers headers hexagon-resource-headers hip-resource-headers hlsl-resource-headers hmaptool libraries llvm-headers llvm-tblgen loongarch-resource-headers mips-resource-headers opencl-resource-headers openmp-resource-headers opt-viewer ppc-htm-resource-headers ppc-resource-headers riscv-resource-headers systemz-resource-headers tests utility-resource-headers ve-resource-headers webassembly-resource-headers windows-resource-headers x86-resource-headers)
set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "ROOT Applications")
set(CPACK_COMPONENT_APPLICATIONS_DESCRIPTION "ROOT executables such as root.exe")
set(CPACK_COMPONENT_APPLICATIONS_INSTALL_TYPES full minimal developer)

# Configuration for component "libraries"

SET(CPACK_COMPONENTS_ALL LLVMAggressiveInstCombine LLVMAnalysis LLVMAsmParser LLVMAsmPrinter LLVMBinaryFormat LLVMBitReader LLVMBitWriter LLVMBitstreamReader LLVMCFGuard LLVMCFIVerify LLVMCodeGen LLVMCodeGenTypes LLVMCore LLVMCoroutines LLVMCoverage LLVMDWARFLinker LLVMDWARFLinkerClassic LLVMDWARFLinkerParallel LLVMDWP LLVMDebugInfoBTF LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoGSYM LLVMDebugInfoLogicalView LLVMDebugInfoMSF LLVMDebugInfoPDB LLVMDebuginfod LLVMDemangle LLVMDiff LLVMDlltoolDriver LLVMExecutionEngine LLVMExegesis LLVMExegesisX86 LLVMExtensions LLVMFileCheck LLVMFrontendDriver LLVMFrontendHLSL LLVMFrontendOffloading LLVMFrontendOpenACC LLVMFrontendOpenMP LLVMFuzzMutate LLVMFuzzerCLI LLVMGlobalISel LLVMHipStdPar LLVMIRPrinter LLVMIRReader LLVMInstCombine LLVMInstrumentation LLVMInterfaceStub LLVMInterpreter LLVMJITLink LLVMLTO LLVMLibDriver LLVMLineEditor LLVMLinker LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCJIT LLVMMCParser LLVMMIRParser LLVMNVPTXCodeGen LLVMNVPTXDesc LLVMNVPTXInfo LLVMObjCARCOpts LLVMObjCopy LLVMObject LLVMObjectYAML LLVMOption LLVMOrcDebugging LLVMOrcJIT LLVMOrcShared LLVMOrcTargetProcess LLVMPasses LLVMProfileData LLVMRemarks LLVMRuntimeDyld LLVMScalarOpts LLVMSelectionDAG LLVMSupport LLVMSymbolize LLVMTableGen LLVMTableGenCommon LLVMTableGenGlobalISel LLVMTarget LLVMTargetParser LLVMTextAPI LLVMTextAPIBinaryReader LLVMTransformUtils LLVMVectorize LLVMWindowsDriver LLVMWindowsManifest LLVMX86AsmParser LLVMX86CodeGen LLVMX86Desc LLVMX86Disassembler LLVMX86Info LLVMX86TargetMCA LLVMXRay LLVMipo LTO Remarks Unspecified aarch64-resource-headers applications arm-common-resource-headers arm-resource-headers bash-autocomplete clang clang-cmake-exports clang-cpp clang-headers clang-resource-headers clang-tblgen clangAPINotes clangAST clangASTMatchers clangAnalysis clangAnalysisFlowSensitive clangAnalysisFlowSensitiveModels clangBasic clangCodeGen clangCrossTU clangDependencyScanning clangDirectoryWatcher clangDriver clangDynamicASTMatchers clangEdit clangExtractAPI clangFormat clangFrontend clangFrontendTool clangIndex clangIndexSerialization clangInterpreter clangLex clangParse clangRewrite clangRewriteFrontend clangSema clangSerialization clangStaticAnalyzerCheckers clangStaticAnalyzerCore clangStaticAnalyzerFrontend clangSupport clangTooling clangToolingASTDiff clangToolingCore clangToolingInclusions clangToolingInclusionsStdlib clangToolingRefactoring clangToolingSyntax clangTransformer cling-cmake-exports clingInterpreter clingMetaProcessor clingUtils cmake-exports core-resource-headers cuda-resource-headers headers hexagon-resource-headers hip-resource-headers hlsl-resource-headers hmaptool libraries llvm-headers llvm-tblgen loongarch-resource-headers mips-resource-headers opencl-resource-headers openmp-resource-headers opt-viewer ppc-htm-resource-headers ppc-resource-headers riscv-resource-headers systemz-resource-headers tests utility-resource-headers ve-resource-headers webassembly-resource-headers windows-resource-headers x86-resource-headers)
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "ROOT Libraries")
set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "All ROOT libraries and dictionaries")
set(CPACK_COMPONENT_LIBRARIES_INSTALL_TYPES full minimal developer)

# Configuration for component "headers"

SET(CPACK_COMPONENTS_ALL LLVMAggressiveInstCombine LLVMAnalysis LLVMAsmParser LLVMAsmPrinter LLVMBinaryFormat LLVMBitReader LLVMBitWriter LLVMBitstreamReader LLVMCFGuard LLVMCFIVerify LLVMCodeGen LLVMCodeGenTypes LLVMCore LLVMCoroutines LLVMCoverage LLVMDWARFLinker LLVMDWARFLinkerClassic LLVMDWARFLinkerParallel LLVMDWP LLVMDebugInfoBTF LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoGSYM LLVMDebugInfoLogicalView LLVMDebugInfoMSF LLVMDebugInfoPDB LLVMDebuginfod LLVMDemangle LLVMDiff LLVMDlltoolDriver LLVMExecutionEngine LLVMExegesis LLVMExegesisX86 LLVMExtensions LLVMFileCheck LLVMFrontendDriver LLVMFrontendHLSL LLVMFrontendOffloading LLVMFrontendOpenACC LLVMFrontendOpenMP LLVMFuzzMutate LLVMFuzzerCLI LLVMGlobalISel LLVMHipStdPar LLVMIRPrinter LLVMIRReader LLVMInstCombine LLVMInstrumentation LLVMInterfaceStub LLVMInterpreter LLVMJITLink LLVMLTO LLVMLibDriver LLVMLineEditor LLVMLinker LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCJIT LLVMMCParser LLVMMIRParser LLVMNVPTXCodeGen LLVMNVPTXDesc LLVMNVPTXInfo LLVMObjCARCOpts LLVMObjCopy LLVMObject LLVMObjectYAML LLVMOption LLVMOrcDebugging LLVMOrcJIT LLVMOrcShared LLVMOrcTargetProcess LLVMPasses LLVMProfileData LLVMRemarks LLVMRuntimeDyld LLVMScalarOpts LLVMSelectionDAG LLVMSupport LLVMSymbolize LLVMTableGen LLVMTableGenCommon LLVMTableGenGlobalISel LLVMTarget LLVMTargetParser LLVMTextAPI LLVMTextAPIBinaryReader LLVMTransformUtils LLVMVectorize LLVMWindowsDriver LLVMWindowsManifest LLVMX86AsmParser LLVMX86CodeGen LLVMX86Desc LLVMX86Disassembler LLVMX86Info LLVMX86TargetMCA LLVMXRay LLVMipo LTO Remarks Unspecified aarch64-resource-headers applications arm-common-resource-headers arm-resource-headers bash-autocomplete clang clang-cmake-exports clang-cpp clang-headers clang-resource-headers clang-tblgen clangAPINotes clangAST clangASTMatchers clangAnalysis clangAnalysisFlowSensitive clangAnalysisFlowSensitiveModels clangBasic clangCodeGen clangCrossTU clangDependencyScanning clangDirectoryWatcher clangDriver clangDynamicASTMatchers clangEdit clangExtractAPI clangFormat clangFrontend clangFrontendTool clangIndex clangIndexSerialization clangInterpreter clangLex clangParse clangRewrite clangRewriteFrontend clangSema clangSerialization clangStaticAnalyzerCheckers clangStaticAnalyzerCore clangStaticAnalyzerFrontend clangSupport clangTooling clangToolingASTDiff clangToolingCore clangToolingInclusions clangToolingInclusionsStdlib clangToolingRefactoring clangToolingSyntax clangTransformer cling-cmake-exports clingInterpreter clingMetaProcessor clingUtils cmake-exports core-resource-headers cuda-resource-headers headers hexagon-resource-headers hip-resource-headers hlsl-resource-headers hmaptool libraries llvm-headers llvm-tblgen loongarch-resource-headers mips-resource-headers opencl-resource-headers openmp-resource-headers opt-viewer ppc-htm-resource-headers ppc-resource-headers riscv-resource-headers systemz-resource-headers tests utility-resource-headers ve-resource-headers webassembly-resource-headers windows-resource-headers x86-resource-headers)
set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C++ Headers")
set(CPACK_COMPONENT_HEADERS_DESCRIPTION "These are needed to do any development")
set(CPACK_COMPONENT_HEADERS_INSTALL_TYPES full developer)

# Configuration for component "tests"

SET(CPACK_COMPONENTS_ALL LLVMAggressiveInstCombine LLVMAnalysis LLVMAsmParser LLVMAsmPrinter LLVMBinaryFormat LLVMBitReader LLVMBitWriter LLVMBitstreamReader LLVMCFGuard LLVMCFIVerify LLVMCodeGen LLVMCodeGenTypes LLVMCore LLVMCoroutines LLVMCoverage LLVMDWARFLinker LLVMDWARFLinkerClassic LLVMDWARFLinkerParallel LLVMDWP LLVMDebugInfoBTF LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoGSYM LLVMDebugInfoLogicalView LLVMDebugInfoMSF LLVMDebugInfoPDB LLVMDebuginfod LLVMDemangle LLVMDiff LLVMDlltoolDriver LLVMExecutionEngine LLVMExegesis LLVMExegesisX86 LLVMExtensions LLVMFileCheck LLVMFrontendDriver LLVMFrontendHLSL LLVMFrontendOffloading LLVMFrontendOpenACC LLVMFrontendOpenMP LLVMFuzzMutate LLVMFuzzerCLI LLVMGlobalISel LLVMHipStdPar LLVMIRPrinter LLVMIRReader LLVMInstCombine LLVMInstrumentation LLVMInterfaceStub LLVMInterpreter LLVMJITLink LLVMLTO LLVMLibDriver LLVMLineEditor LLVMLinker LLVMMC LLVMMCA LLVMMCDisassembler LLVMMCJIT LLVMMCParser LLVMMIRParser LLVMNVPTXCodeGen LLVMNVPTXDesc LLVMNVPTXInfo LLVMObjCARCOpts LLVMObjCopy LLVMObject LLVMObjectYAML LLVMOption LLVMOrcDebugging LLVMOrcJIT LLVMOrcShared LLVMOrcTargetProcess LLVMPasses LLVMProfileData LLVMRemarks LLVMRuntimeDyld LLVMScalarOpts LLVMSelectionDAG LLVMSupport LLVMSymbolize LLVMTableGen LLVMTableGenCommon LLVMTableGenGlobalISel LLVMTarget LLVMTargetParser LLVMTextAPI LLVMTextAPIBinaryReader LLVMTransformUtils LLVMVectorize LLVMWindowsDriver LLVMWindowsManifest LLVMX86AsmParser LLVMX86CodeGen LLVMX86Desc LLVMX86Disassembler LLVMX86Info LLVMX86TargetMCA LLVMXRay LLVMipo LTO Remarks Unspecified aarch64-resource-headers applications arm-common-resource-headers arm-resource-headers bash-autocomplete clang clang-cmake-exports clang-cpp clang-headers clang-resource-headers clang-tblgen clangAPINotes clangAST clangASTMatchers clangAnalysis clangAnalysisFlowSensitive clangAnalysisFlowSensitiveModels clangBasic clangCodeGen clangCrossTU clangDependencyScanning clangDirectoryWatcher clangDriver clangDynamicASTMatchers clangEdit clangExtractAPI clangFormat clangFrontend clangFrontendTool clangIndex clangIndexSerialization clangInterpreter clangLex clangParse clangRewrite clangRewriteFrontend clangSema clangSerialization clangStaticAnalyzerCheckers clangStaticAnalyzerCore clangStaticAnalyzerFrontend clangSupport clangTooling clangToolingASTDiff clangToolingCore clangToolingInclusions clangToolingInclusionsStdlib clangToolingRefactoring clangToolingSyntax clangTransformer cling-cmake-exports clingInterpreter clingMetaProcessor clingUtils cmake-exports core-resource-headers cuda-resource-headers headers hexagon-resource-headers hip-resource-headers hlsl-resource-headers hmaptool libraries llvm-headers llvm-tblgen loongarch-resource-headers mips-resource-headers opencl-resource-headers openmp-resource-headers opt-viewer ppc-htm-resource-headers ppc-resource-headers riscv-resource-headers systemz-resource-headers tests utility-resource-headers ve-resource-headers webassembly-resource-headers windows-resource-headers x86-resource-headers)
set(CPACK_COMPONENT_TESTS_DISPLAY_NAME "ROOT Tests and Tutorials")
set(CPACK_COMPONENT_TESTS_DESCRIPTION "These are needed to do any test and tutorial")
set(CPACK_COMPONENT_TESTS_INSTALL_TYPES full developer)
