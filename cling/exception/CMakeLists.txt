if(NOT MSVC OR win_broken_tests)
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64.*|x86.*|amd64.*|AMD64.*|i686.*|i386.*")
    # All platforms except of ARM/AARCH64
    ROOTTEST_ADD_TEST(nullderef-macro
                      MACRO nullderef.C
                      PASSRC 1
                      LABELS roottest regression cling)
  endif()
endif()

if(NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64) OR M1_BROKEN_TESTS)
  ROOTTEST_ADD_TEST(nullderef-e
                    COMMAND ${ROOT_root_CMD} -l -b -q -e "int*p=nullptr" -e "*p"
                    PASSRC 1
                    LABELS roottest regression cling)
endif()
