file(REMOVE_RECURSE
  "../../../../../lib/libclangTransformer.a"
  "../../../../../lib/libclangTransformer.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangTransformer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
