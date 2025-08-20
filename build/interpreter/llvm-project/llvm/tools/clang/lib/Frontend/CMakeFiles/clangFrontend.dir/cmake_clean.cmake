file(REMOVE_RECURSE
  "../../../../lib/libclangFrontend.a"
  "../../../../lib/libclangFrontend.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangFrontend.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
