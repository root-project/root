foreach(name "atlas" "lhcb" "cms")

  if(name STREQUAL "atlas")
    set(filename "atlasFlushed.root")
    set(_willfail_nolib WILLFAIL)
  elseif(name STREQUAL "cms")
    set(filename "cmsflush.root")
    # has problem with generation library
    set(_willfail WILLFAIL)
  else()
    set(filename "lhcb2.root")
    set(_willfail WILLFAIL)
  endif()

  # all these tests failing and taking extreme long time
  # therefore disable them from the begin
  if(very_long_tests)
    ROOTTEST_ADD_TEST(${name}
                      MACRO readfile.C
                      MACROARG "\"${filename}\",\"\""
                      LABELS longtest
                      ${_willfail}
                      FIXTURES_SETUP root-io-perf-userdatasets-${name}-fixture)

    ROOTTEST_ADD_TEST(${name}-nolib
                      MACRO readfile.C
                      MACROARG "\"${filename}\",\"nolib\""
                      LABELS longtest
                      ${_willfail_nolib} ${_willfail}
                      FIXTURES_REQUIRED root-io-perf-userdatasets-${name}-fixture)

    ROOTTEST_ADD_TEST(${name}-genreflex
                      MACRO readfile.C
                      MACROARG "\"${filename}\",\"genreflex\""
                      LABELS longtest
                      ${_willfail}
                      FIXTURES_REQUIRED root-io-perf-userdatasets-${name}-fixture)
  endif()

endforeach()