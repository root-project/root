file(REMOVE_RECURSE
  "../../../../lib/libclangAnalysis.a"
  "../../../../lib/libclangAnalysis.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangAnalysis.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
