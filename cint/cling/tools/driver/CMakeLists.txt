set( LLVM_USED_LIBS
  libcling)

add_cling_executable(cling
  cling.cpp)

install(TARGETS cling
  RUNTIME DESTINATION bin)
