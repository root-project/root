ROOT_GENERATE_DICTIONARY(G__IoCompressionGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/Event.h
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/EventLinkDef.h)

ROOT_LINKER_LIBRARY(IoCompressionGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/IoCompressionGeneration.cxx
  G__IoCompressionGeneration.cxx
  LIBRARIES Core Tree Hist MathCore Physics Graf Matrix)

if(MSVC)
  add_custom_command(TARGET IoCompressionGeneration POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libIoCompressionGeneration.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libIoCompressionGeneration.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libIoCompressionGeneration.lib
                                     ${CMAKE_CURRENT_BINARY_DIR}/libIoCompressionGeneration.lib)
endif()

ROOTTEST_GENERATE_EXECUTABLE(IoCompressionGenerator
  ${CMAKE_CURRENT_SOURCE_DIR}/IoCompressionGenerator.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore IoCompressionGeneration
  DEPENDS IoCompressionGeneration
  FIXTURES_SETUP io-compression-generator)

set(size 20)
set(opt 1)

function(IO_COMPRESSION_GENERATION comp action)
  ROOTTEST_ADD_TEST("${comp}-${action}"
    COMMAND ./IoCompressionGenerator ${size} "${comp}" ${opt} "${action}"
    FIXTURES_REQUIRED io-compression-generator
    RESOURCE_LOCK io-compression-file)
endfunction()

IO_COMPRESSION_GENERATION(0 1)
IO_COMPRESSION_GENERATION(0 20)
IO_COMPRESSION_GENERATION(5 1)
IO_COMPRESSION_GENERATION(5 20)
IO_COMPRESSION_GENERATION(101 1)
IO_COMPRESSION_GENERATION(101 20)
IO_COMPRESSION_GENERATION(208 1)
IO_COMPRESSION_GENERATION(208 20)
IO_COMPRESSION_GENERATION(301 1)
IO_COMPRESSION_GENERATION(301 20)
IO_COMPRESSION_GENERATION(404 1)
IO_COMPRESSION_GENERATION(404 20)
IO_COMPRESSION_GENERATION(505 1)
IO_COMPRESSION_GENERATION(505 20)
