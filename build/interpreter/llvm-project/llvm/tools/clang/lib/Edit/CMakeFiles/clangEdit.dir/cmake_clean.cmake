file(REMOVE_RECURSE
  "../../../../lib/libclangEdit.a"
  "../../../../lib/libclangEdit.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangEdit.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
