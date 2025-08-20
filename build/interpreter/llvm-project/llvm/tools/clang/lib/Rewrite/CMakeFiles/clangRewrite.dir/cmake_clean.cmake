file(REMOVE_RECURSE
  "../../../../lib/libclangRewrite.a"
  "../../../../lib/libclangRewrite.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangRewrite.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
