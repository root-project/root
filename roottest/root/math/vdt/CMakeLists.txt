if (vdt AND NOT ROOT_CLASSIC_BUILD AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

  set (testname stressVdt)
  set(additional_compile_flags "-O3")

  include_directories(${VDT_INCLUDE_DIRS})

  ROOTTEST_GENERATE_EXECUTABLE(${testname}
                               ${testname}.cxx
                               COMPILE_FLAGS ${additional_compile_flags}
                               LIBRARIES Core Hist Gpad MathCore ${VDT_LIBRARIES}
                               FIXTURES_SETUP root-math-vdt-stressVdt-fixture)

  ROOTTEST_ADD_TEST(${testname}
                    EXEC ./${testname}
                    FAILREGEX "too inaccurate"
                    FIXTURES_REQUIRED root-math-vdt-stressVdt-fixture
                    LABELS longtest)
endif()
