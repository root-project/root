ROOTTEST_GENERATE_DICTIONARY(IoCompressionDict Event.h
  LINKDEF EventLinkDef.h
  FIXTURES_SETUP io-compression-dict)

# The libgen library target is automatically created by ROOTTEST_GENERATE_DICTIONARY
target_sources(IoCompressionDictlibgen PRIVATE Event.cxx)

# For TLorentzVector
target_link_libraries(IoCompressionDictlibgen Physics)

ROOTTEST_GENERATE_EXECUTABLE(IoCompressionGenerator
  MainEvent.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore IoCompressionDictlibgen
  FIXTURES_REQUIRED io-compression-dict
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
