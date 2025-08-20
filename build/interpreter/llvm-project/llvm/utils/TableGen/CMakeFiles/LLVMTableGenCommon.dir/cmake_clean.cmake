file(REMOVE_RECURSE
  "../../lib/libLLVMTableGenCommon.a"
  "../../lib/libLLVMTableGenCommon.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/LLVMTableGenCommon.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
