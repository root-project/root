file(REMOVE_RECURSE
  "../../../../lib/libclangCodeGen.a"
  "../../../../lib/libclangCodeGen.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangCodeGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
