file(REMOVE_RECURSE
  "../../../../lib/libclangAST.a"
  "../../../../lib/libclangAST.pdb"
  "AttrDocTable.inc"
  "Opcodes.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/clangAST.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
