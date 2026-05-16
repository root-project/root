# Locates the ORC runtime archive (compiler-rt's `orc_rt`) and the
# `llvm-jitlink-executor` binary inside the LLVM tree we are linking
# against. Used to bundle them into CppInterOp's distribution so the
# OOP-JIT path works without an external `compiler-rt` install.
#
# Expects find_package(LLVM) to have run already (provides the input
# variables below).
#
# Inputs:
#   LLVM_LIBRARY_DIR
#   LLVM_TOOLS_BINARY_DIR
#   LLVM_VERSION_MAJOR
#
# Output cache variables:
#   LLVMOrcRuntime_FOUND
#   LLVMOrcRuntime_LIBRARY    Path to libclang_rt.orc_rt*.a / liborc_rt*.a
#   LLVMOrcRuntime_EXECUTOR   Path to llvm-jitlink-executor

include(FindPackageHandleStandardArgs)

# compiler-rt installs into `clang/<ver>/lib/<triple>/`; the triple
# subdir is host-derived (`darwin`, `x86_64-unknown-linux-gnu`, etc.).
# Enumerate them at configure time rather than hard-coding the triple.
file(GLOB _orc_hint_dirs LIST_DIRECTORIES TRUE
  "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}/lib/*")

# NAMES covers the spread of compiler-rt archive names across versions
# and platforms: `libclang_rt.orc_rt.a` (modern, per-triple subdir),
# `liborc_rt.a` (legacy unsuffixed), and the arch/OS-suffixed variants
# `_osx` (macOS universal), `-x86_64`, `-aarch64`. NO_DEFAULT_PATH so
# we never pick up a system-installed orc_rt with a different ABI than
# the LLVM we're linking against.
find_library(LLVMOrcRuntime_LIBRARY
  NAMES
    clang_rt.orc_rt
    orc_rt
    orc_rt_osx
    orc_rt-x86_64
    orc_rt-aarch64
  HINTS ${_orc_hint_dirs}
  NO_DEFAULT_PATH)

find_program(LLVMOrcRuntime_EXECUTOR
  NAMES llvm-jitlink-executor
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  NO_DEFAULT_PATH)

find_package_handle_standard_args(LLVMOrcRuntime
  REQUIRED_VARS LLVMOrcRuntime_LIBRARY LLVMOrcRuntime_EXECUTOR)

mark_as_advanced(LLVMOrcRuntime_LIBRARY LLVMOrcRuntime_EXECUTOR)
