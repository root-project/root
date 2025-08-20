file(REMOVE_RECURSE
  "../../../../../../lib/libclangToolingInclusionsStdlib.a"
  "../../../../../../lib/libclangToolingInclusionsStdlib.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangToolingInclusionsStdlib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
