add_llvm_component_library(LLVMLccrt
  Lccrt.cpp
  LccrtDbg.cpp
  LccrtDbgEtydes.cpp
  LccrtDbglines.cpp
  LccrtDbglinesEtydes.cpp
  LccrtEmitter.cpp
  LccrtIpa.cpp
  LccrtIpaPass.cpp
  LccrtMCEmitter.cpp

  LINK_COMPONENTS
  Analysis
  Core
  TransformUtils
  )

if (LLVM_WITH_LCCRT)

include_directories(${LLVM_LCCRT_IDIR})

target_link_directories(LLVMLccrt PUBLIC ${LLVM_LCCRT_LDIR}/)    
target_link_libraries(LLVMLccrt PUBLIC lccrt)

endif()

