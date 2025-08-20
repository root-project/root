file(REMOVE_RECURSE
  "../../../../lib/libclingMetaProcessor.a"
  "../../../../lib/libclingMetaProcessor.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clingMetaProcessor.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
