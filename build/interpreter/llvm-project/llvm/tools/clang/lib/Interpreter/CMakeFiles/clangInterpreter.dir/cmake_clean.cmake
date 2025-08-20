file(REMOVE_RECURSE
  "../../../../lib/libclangInterpreter.a"
  "../../../../lib/libclangInterpreter.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangInterpreter.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
