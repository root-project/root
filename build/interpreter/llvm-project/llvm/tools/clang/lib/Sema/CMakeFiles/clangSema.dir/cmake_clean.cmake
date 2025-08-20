file(REMOVE_RECURSE
  "../../../../lib/libclangSema.a"
  "../../../../lib/libclangSema.pdb"
  "OpenCLBuiltins.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangSema.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
