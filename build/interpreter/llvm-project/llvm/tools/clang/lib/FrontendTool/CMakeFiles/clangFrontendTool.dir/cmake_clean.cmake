file(REMOVE_RECURSE
  "../../../../lib/libclangFrontendTool.a"
  "../../../../lib/libclangFrontendTool.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangFrontendTool.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
