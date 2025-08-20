file(REMOVE_RECURSE
  "../../../../lib/libclangIndex.a"
  "../../../../lib/libclangIndex.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangIndex.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
