if(MSVC AND NOT win_broken_tests)
   return()
endif()

add_library(ROOT10426 SHARED
   ROOT10426.cxx)
target_link_libraries(ROOT10426 PUBLIC ROOT::Core)

ROOTTEST_ADD_TEST(ROOT-10426
   MACRO  execROOT10426.C
   OUTREF execROOT10426.ref)
