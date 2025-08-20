file(REMOVE_RECURSE
  "../../../../lib/libclangFormat.a"
  "../../../../lib/libclangFormat.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangFormat.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
