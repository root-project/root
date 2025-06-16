add_llvm_component_group(Elbrus HAS_JIT)

set(LLVM_TARGET_DEFINITIONS Elbrus.td)

tablegen(LLVM ElbrusGenRegisterInfo.inc -gen-register-info)

add_public_tablegen_target(ElbrusCommonTableGen)

add_llvm_target(ElbrusCodeGen
    ElbrusAsmPrinter.cpp
    ElbrusISelLowering.cpp
    ElbrusFrameLowering.cpp
    ElbrusSubtarget.cpp
    ElbrusTargetMachine.cpp
    ElbrusTargetObjectFile.cpp
    ElbrusTargetTransformInfo.cpp

    LINK_COMPONENTS
    Analysis
    AsmPrinter
    CodeGen
    Core
    ElbrusDesc
    ElbrusInfo
    SelectionDAG
    Support
    Target

    ADD_TO_COMPONENT
    Elbrus
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)

