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

  find_package(Python3 3.8 REQUIRED)
  # Need to duplicate the import ROOT test because the PASS_REGULAR_EXPRESSION
  # property will disable checking the exit code of the test so import ROOT
  # might fail but ctest would not report it
  add_test(NAME python-import-root
    COMMAND ${Python3_EXECUTABLE} -c "import ROOT"
  )
  set_tests_properties(python-import-root
    PROPERTIES ENVIRONMENT PYTHONPATH=${ROOT_LIBRARY_DIR}
  )
  add_test(NAME python-create-histo
    COMMAND ${Python3_EXECUTABLE} -c "import ROOT; h = ROOT.TH1D(\"myHistoName\",\"h\", 100, -5, 5); print(h.GetName());"
  )
  set_tests_properties(python-create-histo
    PROPERTIES ENVIRONMENT PYTHONPATH=${ROOT_LIBRARY_DIR}
               PASS_REGULAR_EXPRESSION "myHistoName"
  )
endif()
