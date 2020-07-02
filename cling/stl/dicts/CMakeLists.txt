ROOT_GENERATE_DICTIONARY(MyDict1 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass1.h)
ROOT_GENERATE_DICTIONARY(MyDict2 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass2.h)
ROOT_GENERATE_DICTIONARY(MyDict3 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass3.h)

ROOT_LINKER_LIBRARY(stldictTest TEST MyClass1.cpp MyClass2.cpp MyClass3.cpp
                                     MyDict1.cxx MyDict2.cxx MyDict3.cxx
                                LIBRARIES ${ROOT_LIBRARIES})

ROOT_ADD_TEST(roottest-cling-stl-dicts-build 
              COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target stldictTest${fast})
