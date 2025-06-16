add_llvm_component_library(LLVMElbrusDesc
  ElbrusMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  ElbrusInfo
  Support

  ADD_TO_COMPONENT
  Elbrus
  )
