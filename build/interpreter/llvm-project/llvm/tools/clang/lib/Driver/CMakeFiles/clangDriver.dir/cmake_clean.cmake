file(REMOVE_RECURSE
  "../../../../lib/libclangDriver.a"
  "../../../../lib/libclangDriver.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangDriver.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
