cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(PostInstall)

find_package(ROOT REQUIRED)

add_executable(readHSimple readHSimple.cxx)
target_link_libraries(readHSimple PUBLIC ROOT::Hist ROOT::RIO)

include(CTest)

if(BUILD_TESTING)
  add_test(NAME run-hsimple
    COMMAND ${ROOT_BINDIR}/root.exe -b -q -l -e ".x ${CMAKE_SOURCE_DIR}/../../tutorials/hsimple.C" -e "return"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  add_test(NAME read-hsimple
    COMMAND $<TARGET_FILE:readHSimple>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(run-hsimple   PROPERTIES FIXTURES_SETUP    HSIMPLE)
  set_tests_properties(read-hsimple  PROPERTIES FIXTURES_REQUIRED HSIMPLE)
endif()
