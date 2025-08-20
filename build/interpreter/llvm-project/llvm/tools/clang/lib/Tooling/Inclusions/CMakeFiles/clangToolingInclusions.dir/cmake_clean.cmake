file(REMOVE_RECURSE
  "../../../../../lib/libclangToolingInclusions.a"
  "../../../../../lib/libclangToolingInclusions.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangToolingInclusions.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
