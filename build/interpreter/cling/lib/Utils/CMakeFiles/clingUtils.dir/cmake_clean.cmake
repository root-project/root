file(REMOVE_RECURSE
  "../../../../lib/libclingUtils.a"
  "../../../../lib/libclingUtils.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clingUtils.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
