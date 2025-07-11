if(experimental_adaptivecpp)
  ROOTTEST_ADD_TEST(syclbasic
                    MACRO syclbasic.C
                    LABELS roottest regression cling sycl)
  set_property(TEST roottest-cling-sycl-syclbasic PROPERTY ENVIRONMENT "ACPP_DEBUG_LEVEL=1")
endif()

