ROOT_GENERATE_DICTIONARY(MyDict1 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass1.h)
ROOT_GENERATE_DICTIONARY(MyDict2 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass2.h)
ROOT_GENERATE_DICTIONARY(MyDict3 ${CMAKE_CURRENT_SOURCE_DIR}/MyClass3.h)

# This is a compile-time test
ROOT_LINKER_LIBRARY(stldictTest NOINSTALL MyClass1.cpp MyClass2.cpp MyClass3.cpp
                    MyDict1.cxx MyDict2.cxx MyDict3.cxx
                    LIBRARIES ROOT::Core)
ROOT_SET_OUTPUT_DIRECTORIES(stldictTest)
# With MSVC, roottest dictionaries don't depend on rootcling (see ROOT_GENERATE_DICTIONARY and 5d86d90d487)
add_dependencies(MyDict1 rootcling)
add_dependencies(MyDict2 rootcling)
add_dependencies(MyDict3 rootcling)
add_dependencies(stldictTest MyDict1 MyDict2 MyDict3)
