foreach(comp 0 101 208 301 404 505 5)

  set(file_name Event_${comp}.root)

  ROOTTEST_ADD_TEST(write-${comp}
                    COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/bigevent/IoBigEventGenerator 20 ${comp} 1 1 600 ${file_name}
                    FIXTURES_REQUIRED root-io-bigevent-generator-fixture
                    FIXTURES_SETUP root-io-compression-write-${comp}-fixture)

  ROOTTEST_ADD_TEST(read-${comp}
                    COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/bigevent/IoBigEventGenerator 20 ${comp} 1 20 600 ${file_name}
                    FIXTURES_REQUIRED root-io-compression-write-${comp}-fixture)
endforeach()