file(REMOVE_RECURSE
  "../../../../lib/libclangBasic.a"
  "../../../../lib/libclangBasic.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangBasic.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
