file(REMOVE_RECURSE
  "../../../../../lib/libclangDependencyScanning.a"
  "../../../../../lib/libclangDependencyScanning.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangDependencyScanning.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
