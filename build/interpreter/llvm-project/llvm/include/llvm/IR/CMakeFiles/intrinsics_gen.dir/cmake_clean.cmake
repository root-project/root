file(REMOVE_RECURSE
  "Attributes.inc"
  "CMakeFiles/intrinsics_gen"
  "IntrinsicEnums.inc"
  "IntrinsicImpl.inc"
  "IntrinsicsAArch64.h"
  "IntrinsicsAMDGPU.h"
  "IntrinsicsARM.h"
  "IntrinsicsBPF.h"
  "IntrinsicsDirectX.h"
  "IntrinsicsHexagon.h"
  "IntrinsicsLoongArch.h"
  "IntrinsicsMips.h"
  "IntrinsicsNVPTX.h"
  "IntrinsicsPowerPC.h"
  "IntrinsicsR600.h"
  "IntrinsicsRISCV.h"
  "IntrinsicsS390.h"
  "IntrinsicsSPIRV.h"
  "IntrinsicsVE.h"
  "IntrinsicsWebAssembly.h"
  "IntrinsicsX86.h"
  "IntrinsicsXCore.h"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/intrinsics_gen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
