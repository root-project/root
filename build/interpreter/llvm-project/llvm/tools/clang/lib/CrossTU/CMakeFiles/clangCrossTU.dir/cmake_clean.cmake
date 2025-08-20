file(REMOVE_RECURSE
  "../../../../lib/libclangCrossTU.a"
  "../../../../lib/libclangCrossTU.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangCrossTU.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
