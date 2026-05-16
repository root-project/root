set(LLVM_LINK_COMPONENTS Support TableGen)

add_tablegen(cppinterop-tblgen CPPINTEROP
  CppInterOpEmitter.cpp
  TableGen.cpp
)
