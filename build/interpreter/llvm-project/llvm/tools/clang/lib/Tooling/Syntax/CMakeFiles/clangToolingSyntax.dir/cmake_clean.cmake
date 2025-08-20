file(REMOVE_RECURSE
  "../../../../../lib/libclangToolingSyntax.a"
  "../../../../../lib/libclangToolingSyntax.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangToolingSyntax.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
