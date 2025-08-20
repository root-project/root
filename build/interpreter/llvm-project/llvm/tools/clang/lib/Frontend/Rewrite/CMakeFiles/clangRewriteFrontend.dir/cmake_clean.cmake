file(REMOVE_RECURSE
  "../../../../../lib/libclangRewriteFrontend.a"
  "../../../../../lib/libclangRewriteFrontend.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangRewriteFrontend.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
