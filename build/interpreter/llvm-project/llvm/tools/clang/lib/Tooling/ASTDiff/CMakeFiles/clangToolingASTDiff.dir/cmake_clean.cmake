file(REMOVE_RECURSE
  "../../../../../lib/libclangToolingASTDiff.a"
  "../../../../../lib/libclangToolingASTDiff.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangToolingASTDiff.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
