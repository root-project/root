file(REMOVE_RECURSE
  "../../include/clang/Tooling/NodeIntrospection.inc"
  "ASTNodeAPI.json"
  "CMakeFiles/run-ast-api-generate-tool"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/run-ast-api-generate-tool.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
