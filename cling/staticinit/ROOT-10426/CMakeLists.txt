# Recursive jitting totally doesn't work without modules; the parent transaction
# is still active and we cannot run the nested JIT.
# See https://sft.its.cern.ch/jira/browse/ROOT-10463
#
# For modules we load based on the missing symbol, i.e. after the parent
# transaction has been committed.
if(NOT ROOT_runtime_cxxmodules_FOUND)
  return()
endif()

add_library(ROOT10426 SHARED
   ROOT10426.cxx)
target_link_libraries(ROOT10426 PUBLIC Core)

ROOTTEST_ADD_TEST(ROOT-10426
   MACRO  execROOT10426.C
   OUTREF execROOT10426.ref)
