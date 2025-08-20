file(REMOVE_RECURSE
  "../../../../lib/libclangAPINotes.a"
  "../../../../lib/libclangAPINotes.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangAPINotes.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
