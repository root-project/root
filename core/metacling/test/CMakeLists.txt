# Make llvm and clang available here as we are mostly testing methods directly
# depending on them.
include_directories(SYSTEM
  ${CLANG_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CLAD_INCLUDE_DIRS}
)

include_directories(
  ../res
  ../../clingutils/res
  ../../foundation/res
  ${CLING_INCLUDE_DIRS}
)
ROOT_ADD_UNITTEST_DIR(Core RIO LLVMSupport)
