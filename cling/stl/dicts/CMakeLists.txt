ROOT_GENERATE_DICTIONARY(MyDict1 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass1.h)
ROOT_GENERATE_DICTIONARY(MyDict2 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass2.h)
ROOT_GENERATE_DICTIONARY(MyDict3 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass3.h)

ROOT_LINKER_LIBRARY(stldictTest TEST MyClass1.cpp MyClass2.cpp MyClass3.cpp
                                     MyDict1.cxx MyDict2.cxx MyDict3.cxx
                                LIBRARIES ${ROOT_LIBRARIES})

# Right now, this target doesn't seems subject to CMP0113. If this changes in
# the future, add "MyDict1${fast} MyDict2${fast} MyDict3${fast}" to the list
# of targets. Doing so right now would build the dictionaries twice.
ROOT_ADD_TEST(roottest-cling-stl-dicts-build 
              COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target stldictTest${fast} -- ${always-make})
