cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(PostInstall)

find_package(ROOT REQUIRED)

add_executable(readHSimple readHSimple.cxx)
target_link_libraries(readHSimple PUBLIC ROOT::Hist ROOT::RIO)

include(CTest)

if(BUILD_TESTING)
  add_test(NAME run-hsimple
    COMMAND ${ROOT_BINDIR}/root.exe -b -q -e ".x ${CMAKE_SOURCE_DIR}/../../tutorials/hsimple.C" -e "return"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  add_test(NAME read-hsimple
    COMMAND $<TARGET_FILE:readHSimple>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(run-hsimple   PROPERTIES FIXTURES_SETUP    HSIMPLE)
  set_tests_properties(read-hsimple  PROPERTIES FIXTURES_REQUIRED HSIMPLE)

  # Add a test to check root-config gives the right compiler invocation flags
  execute_process(
    COMMAND ${ROOT_BINDIR}/root-config --cflags --libs
    OUTPUT_VARIABLE ROOT_CONFIG_OUTPUT_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Pass a list to the compiler invocation to avoid errors such as "invalid library name" when passing the quoted string
  # output from root-config
  string(REPLACE " " ";" ROOT_CONFIG_OUTPUT "${ROOT_CONFIG_OUTPUT_STR}")
  add_test(NAME read-hsimple-root-config-build
    COMMAND ${CMAKE_CXX_COMPILER} -o readHSimple-root-config ${CMAKE_SOURCE_DIR}/readHSimple.cxx ${ROOT_CONFIG_OUTPUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  add_test(NAME read-hsimple-root-config-run
    COMMAND readHSimple-root-config
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(read-hsimple-root-config-build   PROPERTIES FIXTURES_SETUP    HSIMPLE-ROOT-CONFIG)
  set_tests_properties(read-hsimple-root-config-run     PROPERTIES FIXTURES_REQUIRED "HSIMPLE;HSIMPLE-ROOT-CONFIG")

  if(ROOT_pyroot_FOUND)
    find_package(Python3 3.10 REQUIRED)
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
    add_test(NAME python-package-metadata
      COMMAND ${Python3_EXECUTABLE} -c "import importlib.metadata as meta, ROOT; meta.version(\"ROOT\")"
    )
    set_tests_properties(python-package-metadata
      PROPERTIES ENVIRONMENT PYTHONPATH=${ROOT_LIBRARY_DIR}
    )

    # Test ACLiC can be run from Python after installation
    configure_file(post_install_test_ACLiC.C ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
    configure_file(post_install_test_ACLiC.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
    add_test(NAME python-post-install-ACLiC
      COMMAND ${Python3_EXECUTABLE} post_install_test_ACLiC.py
    )
    set_tests_properties(python-post-install-ACLiC
      PROPERTIES
      ENVIRONMENT PYTHONPATH=${ROOT_LIBRARY_DIR}
      FIXTURES_REQUIRED python-post-install-ACLiC-fixture
    )
    add_test(NAME python-post-install-ACLiC-cleanupfiles
      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/post_install_test_ACLiC_cleanup.cmake
    )
    set_tests_properties(python-post-install-ACLiC-cleanupfiles
      PROPERTIES FIXTURES_CLEANUP python-post-install-ACLiC-fixture
    )

  endif()

  if(TARGET ROOT::ROOTVecOps)
    add_executable(RVec testRVec.cxx)
    target_link_libraries(RVec PUBLIC ROOT::ROOTVecOps)
    add_test(NAME RVec COMMAND $<TARGET_FILE:RVec>)
  endif()
endif()
