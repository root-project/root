file(REMOVE_RECURSE
  "../../../../lib/libclangLex.a"
  "../../../../lib/libclangLex.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangLex.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
