ROOT_GENERATE_DICTIONARY(MyDict1 MyClass1.h)
ROOT_GENERATE_DICTIONARY(MyDict2 MyClass2.h)
ROOT_GENERATE_DICTIONARY(MyDict3 MyClass3.h)

ROOT_LINKER_LIBRARY(stldictTest TEST MyClass1.cpp MyClass2.cpp MyClass3.cpp
                                     MyDict1.cxx MyDict2.cxx MyDict3.cxx
                                LIBRARIES ${ROOT_LIBRARIES})

ROOT_ADD_TEST(roottest-cling-stl-dicts-build 
              COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target stldictTest${fast})
