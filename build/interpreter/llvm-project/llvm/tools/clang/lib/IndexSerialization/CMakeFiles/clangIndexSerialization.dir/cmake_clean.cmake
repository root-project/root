file(REMOVE_RECURSE
  "../../../../lib/libclangIndexSerialization.a"
  "../../../../lib/libclangIndexSerialization.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangIndexSerialization.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
