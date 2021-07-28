set(LLVM_LINK_COMPONENTS
  Core
  MC
  Support
  )

# Cut dependency on git to prevent rebuild of large parts of ROOT.
# This will be solved "properly" in llvm upstream, resurrecting
# https://reviews.llvm.org/rGd452126cb333cbd6b289b01d9bc9a48b111ffe9a
# which got lost in (mono-repo) 23fdd5a37ff4e0512af0b40f6ff3e6db4694e937
# 2019-02-06 03:51 +0000 Petr Hosek
# o [CMake] Unify scripts for generating VCS headers
#
#find_first_existing_vc_file("${LLVM_MAIN_SRC_DIR}" llvm_vc)
#find_first_existing_vc_file("${CLANG_SOURCE_DIR}" clang_vc)

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc")

set(generate_vcs_version_script "${LLVM_CMAKE_PATH}/GenerateVersionFromVCS.cmake")

if(llvm_vc)
  set(llvm_source_dir ${LLVM_MAIN_SRC_DIR})
endif()
if(clang_vc)
  set(clang_source_dir ${CLANG_SOURCE_DIR})
endif()

# Create custom target to generate the VC revision include.
add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${llvm_vc}" "${clang_vc}" "${generate_vcs_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DNAMES=\"LLVM;CLANG\""
                           "-DLLVM_SOURCE_DIR=${llvm_source_dir}"
                           "-DCLANG_SOURCE_DIR=${clang_source_dir}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${generate_vcs_version_script}")

# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

set_property(SOURCE Version.cpp APPEND PROPERTY
             COMPILE_DEFINITIONS "HAVE_VCS_VERSION_INC")

add_clang_library(clangBasic
  Attributes.cpp
  Builtins.cpp
  CharInfo.cpp
  CodeGenOptions.cpp
  Cuda.cpp
  Diagnostic.cpp
  DiagnosticIDs.cpp
  DiagnosticOptions.cpp
  FileManager.cpp
  FileSystemStatCache.cpp
  FixedPoint.cpp
  IdentifierTable.cpp
  LangOptions.cpp
  Module.cpp
  ObjCRuntime.cpp
  OpenMPKinds.cpp
  OperatorPrecedence.cpp
  SanitizerBlacklist.cpp
  SanitizerSpecialCaseList.cpp
  Sanitizers.cpp
  SourceLocation.cpp
  SourceManager.cpp
  TargetInfo.cpp
  Targets.cpp
  Targets/AArch64.cpp
  Targets/AMDGPU.cpp
  Targets/ARC.cpp
  Targets/ARM.cpp
  Targets/AVR.cpp
  Targets/BPF.cpp
  Targets/Hexagon.cpp
  Targets/Lanai.cpp
  Targets/Le64.cpp
  Targets/MSP430.cpp
  Targets/Mips.cpp
  Targets/NVPTX.cpp
  Targets/OSTargets.cpp
  Targets/PNaCl.cpp
  Targets/PPC.cpp
  Targets/RISCV.cpp
  Targets/SPIR.cpp
  Targets/Sparc.cpp
  Targets/SystemZ.cpp
  Targets/TCE.cpp
  Targets/WebAssembly.cpp
  Targets/X86.cpp
  Targets/XCore.cpp
  TokenKinds.cpp
  Version.cpp
  Warnings.cpp
  XRayInstr.cpp
  XRayLists.cpp
  ${version_inc}
  )

# Some versions of Debian/Ubuntu have a buggy LLVM/Clang package.
# We workaround this case by defining a preprocessor symbol.
if (NOT builtin_llvm)
  file(READ ${LLVM_DIR}/include/llvm/ADT/Triple.h TMPTXT)
  string(FIND "${TMPTXT}" "KFreeBSD" matchres)
  if(${matchres} EQUAL -1)
    set_property(SOURCE Targets.cpp APPEND PROPERTY
                 COMPILE_DEFINITIONS "KFreeBSD=kFreeBSD")
  endif ()
endif()