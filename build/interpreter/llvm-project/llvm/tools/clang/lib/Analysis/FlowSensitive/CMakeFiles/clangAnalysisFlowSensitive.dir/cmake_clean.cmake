file(REMOVE_RECURSE
  "../../../../../lib/libclangAnalysisFlowSensitive.a"
  "../../../../../lib/libclangAnalysisFlowSensitive.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangAnalysisFlowSensitive.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
