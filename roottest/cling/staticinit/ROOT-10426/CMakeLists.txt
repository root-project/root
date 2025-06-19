# Recursive jitting totally doesn't work without modules; the parent transaction
# is still active and we cannot run the nested JIT.
# See https://sft.its.cern.ch/jira/browse/ROOT-10463
#
# For modules we load based on the missing symbol, i.e. after the parent
# transaction has been committed.
if(NOT runtime_cxxmodules)
  return()
endif()

add_library(ROOT10426 SHARED ROOT10426.cxx)
target_link_libraries(ROOT10426 PUBLIC ROOT::Core)

add_test(NAME roottest-cling-staticinit-ROOT-10426-lib
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                   ${build_config}
                                   --target ROOT10426${fast}
                                   -- ${always-make})

set_property(TEST roottest-cling-staticinit-ROOT-10426-lib PROPERTY FIXTURES_SETUP cling-staticinit-ROOT-10426-fixture)

ROOTTEST_ADD_TEST(run
                  MACRO  execROOT10426.C
                  OUTREF execROOT10426.ref
                  FIXTURES_REQUIRED cling-staticinit-ROOT-10426-fixture)
