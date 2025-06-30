set(refname "")

if(${compression_default} STREQUAL "lz4")
  set(refname testForeignDrawLZ4.ref)
else(${compression_default} STREQUAL "zlib")
  if(ZLIB_CF)
    set(refname testForeignDrawZLIB_builtinzlib.ref)
  else()
    set(refname testForeignDrawZLIB.ref)
   endif()
endif()

if(NOT refname STREQUAL "")
  ROOTTEST_ADD_TEST(def
                   MACRO def.C+
                   FIXTURES_SETUP root-io-treeForeign-def-fixture)

  ROOTTEST_ADD_TEST(testForeignDraw
                    MACRO Run.C
                    OUTREF ${refname}
                    FIXTURES_REQUIRED root-io-treeForeign-def-fixture)
endif()

