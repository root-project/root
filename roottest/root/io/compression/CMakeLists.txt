ROOTTEST_GENERATE_DICTIONARY(libIoCompressionEvent
  Event.h
  LINKDEF EventLinkDef.h
  SOURCES Event.cxx
  NO_ROOTMAP
  NO_CXXMODULE
  FIXTURES_SETUP io-compression-dict)

target_link_libraries(libIoCompressionEventlibgen Physics)

ROOTTEST_GENERATE_EXECUTABLE(MainEvent
  MainEvent.cxx
  LIBRARIES Core RIO libIoCompressionEventlibgen
  FIXTURES_REQUIRED io-compression-dict
  FIXTURES_SETUP io-compression-exec)

set(values 0 101 208 301 404 505 5)

foreach(comp ${values})
  ROOTTEST_ADD_TEST(write-${comp}
    EXEC ./MainEvent
    OPTS 20 ${comp} 1 1
    FIXTURES_REQUIRED io-compression-exec
    FIXTURES_SETUP io-compression-write-${comp})

  ROOTTEST_ADD_TEST(read-${comp}
    EXEC ./MainEvent
    OPTS 20 ${comp} 1 20
    FIXTURES_REQUIRED io-compression-write-${comp})
endforeach()