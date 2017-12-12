if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
  ROOTTEST_ADD_TEST(nullderef-macro
                    MACRO nullderef.C
                    PASSRC 1
                    LABELS roottest regression cling)
endif()

ROOTTEST_ADD_TEST(nullderef-e
                  COMMAND root -l -b -q -e 'int*p=nullptr;*p'
                  PASSRC 1
                  LABELS roottest regression cling)
