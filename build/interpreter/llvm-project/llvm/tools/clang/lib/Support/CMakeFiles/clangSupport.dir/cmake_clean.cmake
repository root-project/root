file(REMOVE_RECURSE
  "../../../../lib/libclangSupport.a"
  "../../../../lib/libclangSupport.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangSupport.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
