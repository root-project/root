file(REMOVE_RECURSE
  "../../../../lib/libclangExtractAPI.a"
  "../../../../lib/libclangExtractAPI.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangExtractAPI.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
