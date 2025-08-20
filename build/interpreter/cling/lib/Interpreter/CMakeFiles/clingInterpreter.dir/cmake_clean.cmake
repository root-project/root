file(REMOVE_RECURSE
  "../../../../lib/libclingInterpreter.a"
  "../../../../lib/libclingInterpreter.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clingInterpreter.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
