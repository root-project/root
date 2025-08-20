file(REMOVE_RECURSE
  "../../../../../../lib/libclangAnalysisFlowSensitiveModels.a"
  "../../../../../../lib/libclangAnalysisFlowSensitiveModels.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangAnalysisFlowSensitiveModels.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
