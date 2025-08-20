file(REMOVE_RECURSE
  "../../../../../lib/libclangStaticAnalyzerFrontend.a"
  "../../../../../lib/libclangStaticAnalyzerFrontend.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangStaticAnalyzerFrontend.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
