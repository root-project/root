ROOTTEST_ADD_TEST(execKeyOrder
                  MACRO execKeyOrder.C
                  OUTREF execKeyOrder.ref)

ROOTTEST_ADD_TEST(execCreateAndMerge
                  MACRO execCreateAndMerge.C
                  OUTREF references/execCreateAndMerge.ref)

ROOTTEST_ADD_TEST(execMergeMulti
                  MACRO execMergeMulti.C
                  OUTREF references/execMergeMulti.ref
                  FIXTURES_SETUP root-io-filemerger-MergeMulti-fixture)

ROOTTEST_ADD_TEST(execFileMerger
                  MACRO execFileMerger.C
                  OUTREF references/execFileMerger.ref)

ROOTTEST_ADD_TEST(datagen-hadd-mfile12
                  COMMAND ${ROOT_hadd_CMD} -f mfile1-2.root mfile1.root mfile2.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mfile34
                  COMMAND ${ROOT_hadd_CMD} -f mfile3-4.root mfile3.root mfile4.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mzfile12
                  COMMAND ${ROOT_hadd_CMD} -f209 mzfile1-2.root mfile1.root mfile2.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzfile12-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mzfile34
                  COMMAND ${ROOT_hadd_CMD} -f209 mzfile3-4.root mfile3.root mfile4.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzfile34-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mzlibfile12
                  COMMAND ${ROOT_hadd_CMD} -f101 mzlibfile1-2.root mfile1.root mfile2.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzlibfile12-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mzlibfile34
                  COMMAND ${ROOT_hadd_CMD} -f101 mzlibfile3-4.root mfile3.root mfile4.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzlibfile34-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mlz4file12
                  COMMAND ${ROOT_hadd_CMD} -f404 mlz4file1-2.root mfile1.root mfile2.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mlz4file12-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mlz4file34
                  COMMAND ${ROOT_hadd_CMD} -f404 mlz4file3-4.root mfile3.root mfile4.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mlz4file34-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mzstdfile12
                  COMMAND ${ROOT_hadd_CMD} -f505 mzstdfile1-2.root mfile1.root mfile2.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzstdfile12-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mzstdfile34
                  COMMAND ${ROOT_hadd_CMD} -f505 mzstdfile3-4.root mfile3.root mfile4.root
                  FIXTURES_REQUIRED root-io-filemerger-MergeMulti-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzstdfile34-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mzlibfile14
                  COMMAND ${ROOT_hadd_CMD} -f101 mzlibfile1-4.root mzlibfile1-2.root mzlibfile3-4.root
                  FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzlibfile12-fixture
                                    root-io-filemerger-datagen-hadd-mzlibfile34-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzlibfile14-fixture)

if(ZLIB_CF)
  ROOTTEST_ADD_TEST(merged-zlib
                    MACRO testMergedFile.C
                    MACROARG "\"mzlibfile1-4.root\",101,4804,20"
                    FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzlibfile14-fixture)
else()
  ROOTTEST_ADD_TEST(merged-zlib
                    MACRO testMergedFile.C
                    MACROARG "\"mzlibfile1-4.root\",101,5014,20"
                    FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzlibfile14-fixture)
endif()

ROOTTEST_ADD_TEST(datagen-hadd-mzfile14
                  COMMAND ${ROOT_hadd_CMD} -f207 mzfile1-4.root mzfile1-2.root mzfile3-4.root
                  FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzfile12-fixture
                                    root-io-filemerger-datagen-hadd-mzfile34-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzfile14-fixture)

ROOTTEST_ADD_TEST(merged-lzma
                  MACRO testMergedFile.C
                  MACROARG "\"mzfile1-4.root\",207,4455,20"
                  FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzfile14-fixture)

ROOTTEST_ADD_TEST(datagen-hadd-mlz4file14
                  COMMAND ${ROOT_hadd_CMD} -f404 mlz4file1-4.root mlz4file1-2.root mlz4file3-4.root
                  FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mlz4file12-fixture
                                    root-io-filemerger-datagen-hadd-mlz4file34-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mlz4file14-fixture)

if(${LZ4_VERSION} VERSION_LESS "1.7.5")
   ROOTTEST_ADD_TEST(merged-lz4
                     MACRO testMergedFile.C
                     MACROARG "\"mlz4file1-4.root\",404,5422,5"
                     FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mlz4file14-fixture)
elseif(${LZ4_VERSION} VERSION_GREATER_EQUAL "1.7.5")
   ROOTTEST_ADD_TEST(merged-lz4
                     MACRO testMergedFile.C
                     MACROARG "\"mlz4file1-4.root\",404,5395,5"
                     FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mlz4file14-fixture)
endif()

ROOTTEST_ADD_TEST(datagen-hadd-mzstdfile14
                  COMMAND ${ROOT_hadd_CMD} -f505 mzstdfile1-4.root mzstdfile1-2.root mzstdfile3-4.root
                  FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzstdfile12-fixture
                                    root-io-filemerger-datagen-hadd-mzstdfile34-fixture
                  FIXTURES_SETUP root-io-filemerger-datagen-hadd-mzstdfile14-fixture)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
   if(${ZSTD_VERSION} VERSION_LESS_EQUAL "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
                        MACRO testMergedFile.C
                        MACROARG "\"mzstdfile1-4.root\",505,4533,20"
                        FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzstdfile14-fixture)
   else(${ZSTD_VERSION} VERSION_GREATER "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
                        MACRO testMergedFile.C
                        MACROARG "\"mzstdfile1-4.root\",505,4456,20"
                        FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzstdfile14-fixture)
   endif()
else()
   if(${ZSTD_VERSION} VERSION_LESS_EQUAL "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
                        MACRO testMergedFile.C
                        MACROARG "\"mzstdfile1-4.root\",505,4529,20"
                        FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzstdfile14-fixture)
   else(${ZSTD_VERSION} VERSION_GREATER "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
                        MACRO testMergedFile.C
                        MACROARG "\"mzstdfile1-4.root\",505,4454,20"
                        FIXTURES_REQUIRED root-io-filemerger-datagen-hadd-mzstdfile14-fixture)
   endif()
endif()

ROOTTEST_ADD_TEST(hsimple
                  ROOTEXE_OPTS -e "#pragma STDC FP_CONTRACT OFF"
                  MACRO ${ROOT_TUTORIALS_DIR}/hsimple.C
                  PASSRC 255
                  FIXTURES_SETUP root-io-filemerger-hsimple-fixture)

ROOTTEST_ADD_TEST(hsimple209
                  COMMAND ${ROOT_hadd_CMD} -f209 hsimple209.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple209-fixture)

ROOTTEST_ADD_TEST(simple-lzma-compr-level9
                  MACRO testSimpleFile.C
                  MACROARG "\"hsimple209.root\",25000,209,392709,17"
                  FIXTURES_REQUIRED root-io-filemerger-hsimple209-fixture)

# all following tests performed only when zlib is default compression
if(NOT ${compression_default} STREQUAL "zlib")
  return()
endif()

ROOTTEST_ADD_TEST(hsimple9
                  COMMAND ${ROOT_hadd_CMD} -f9 hsimple9.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple9-fixture)

ROOTTEST_ADD_TEST(hsimple9x2-first
                  COMMAND ${ROOT_hadd_CMD} -f9 hsimple9x2.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple9x2-first-fixture)

ROOTTEST_ADD_TEST(hsimple9x2
                  COMMAND ${ROOT_hadd_CMD} -f9 -a hsimple9x2.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple9x2-first-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple9x2-fixture)

ROOTTEST_ADD_TEST(hsimple109x2-first
                  COMMAND ${ROOT_hadd_CMD} -f109 hsimple109x2.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple109x2-first-fixture)

ROOTTEST_ADD_TEST(hsimple109x2
                  COMMAND ${ROOT_hadd_CMD} -f109 -a hsimple109x2.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple109x2-first-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple109x2-fixture)

ROOTTEST_ADD_TEST(hsimple101
                  COMMAND ${ROOT_hadd_CMD} -f101 hsimple101.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple101-fixture)

ROOTTEST_ADD_TEST(hsimple106
                  COMMAND ${ROOT_hadd_CMD} -f106 hsimple106.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple106-fixture)

ROOTTEST_ADD_TEST(hsimple109
                  COMMAND ${ROOT_hadd_CMD} -f109 hsimple109.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple109-fixture)

ROOTTEST_ADD_TEST(hsimple401
                  COMMAND ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple401-fixture)

ROOTTEST_ADD_TEST(hsimple404
                  COMMAND ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple404-fixture)

ROOTTEST_ADD_TEST(hsimple409
                  COMMAND ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimple409-fixture)

ROOTTEST_ADD_TEST(hsimpleK
                  COMMAND ${ROOT_hadd_CMD} -fk hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple9x2.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple209-fixture
                                    root-io-filemerger-hsimple409-fixture
                                    root-io-filemerger-hsimple-fixture
                                    root-io-filemerger-hsimple9-fixture
                                    root-io-filemerger-hsimple9x2-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimpleK-fixture)

ROOTTEST_ADD_TEST(hsimpleK209
                  COMMAND ${ROOT_hadd_CMD} -fk209 hsimpleK209.root hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple9x2.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimpleK-fixture
                                    root-io-filemerger-hsimple209-fixture
                                    root-io-filemerger-hsimple409-fixture
                                    root-io-filemerger-hsimple-fixture
                                    root-io-filemerger-hsimple9-fixture
                                    root-io-filemerger-hsimple9x2-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimpleK209-fixture)

ROOTTEST_ADD_TEST(hsimpleK404
                  COMMAND ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple9x2.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimpleK209-fixture
                                    root-io-filemerger-hsimpleK-fixture
                                    root-io-filemerger-hsimple-fixture
                                    root-io-filemerger-hsimple9-fixture
                                    root-io-filemerger-hsimple209-fixture
                                    root-io-filemerger-hsimple409-fixture
                                    root-io-filemerger-hsimple9x2-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimpleK404-fixture)

ROOTTEST_ADD_TEST(hsimpleF
                  COMMAND ${ROOT_hadd_CMD} -ff hsimpleF.root hsimple9.root hsimple209.root hsimple409.root hsimpleK404.root hsimple.root hsimple9x2.root
                  FIXTURES_REQUIRED root-io-filemerger-hsimple9-fixture
                                    root-io-filemerger-hsimple209-fixture
                                    root-io-filemerger-hsimple409-fixture
                                    root-io-filemerger-hsimpleK404-fixture
                                    root-io-filemerger-hsimple-fixture
                                    root-io-filemerger-hsimple9x2-fixture
                  FIXTURES_SETUP root-io-filemerger-hsimpleF-fixture)

if(ZLIB_CF)
   ROOTTEST_ADD_TEST(simple-default-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple9.root\",25000,9,441500,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple9-fixture)

   ROOTTEST_ADD_TEST(simple-zlib-compr-level1
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple101.root\",25000,101,443022,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple101-fixture)

   ROOTTEST_ADD_TEST(simple-zlib-compr-level6
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple106.root\",25000,106,441608,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple106-fixture)

   ROOTTEST_ADD_TEST(simple-zlib-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple109.root\",25000,109,441508,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple109-fixture)

   ROOTTEST_ADD_TEST(simplex2-default-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple9x2.root\",50000,9,871048,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple9x2-fixture)

   ROOTTEST_ADD_TEST(simplex2-zlib-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple109x2.root\",50000,109,871053,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple109x2-fixture)

   ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple409.root\",25000,409,517008,15"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple409-fixture)

   ROOTTEST_ADD_TEST(simplek-default-compr-deflevel
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimpleK.root\",6*25000,209,2297616,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimpleK-fixture)

   ROOTTEST_ADD_TEST(simplek-lzma-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimpleK209.root\",12*25000,209,4583840,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimpleK209-fixture)

   ROOTTEST_ADD_TEST(simplef-default-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimpleF.root\",30*25000,9,12889798,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimpleF-fixture)

   ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimpleK404.root\",24*25000,404,12047492,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimpleK404-fixture)

   ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple404.root\",25000,404,517262,15"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple404-fixture)

   ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple401.root\",25000,401,519976,5"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple401-fixture)
else()
   ROOTTEST_ADD_TEST(simple-default-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple9.root\",25000,9,431015,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple9-fixture)

   ROOTTEST_ADD_TEST(simple-zlib-compr-level1
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple101.root\",25000,101,415025,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple101-fixture)

   ROOTTEST_ADD_TEST(simple-zlib-compr-level6
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple106.root\",25000,106,431303,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple106-fixture)

   ROOTTEST_ADD_TEST(simple-zlib-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple109.root\",25000,109,431024,25"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple109-fixture)

   ROOTTEST_ADD_TEST(simplex2-default-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple9x2.root\",50000,9,850127,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple9x2-fixture)

   ROOTTEST_ADD_TEST(simplex2-zlib-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimple109x2.root\",50000,109,850136,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimple109x2-fixture)

   ROOTTEST_ADD_TEST(simplek-default-compr-deflevel
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimpleK.root\",6*25000,209,2297616,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimpleK-fixture)

   ROOTTEST_ADD_TEST(simplek-lzma-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimpleK209.root\",12*25000,209,4583840,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimpleK209-fixture)

   if(${LZ4_VERSION} VERSION_LESS "1.7.5")
      ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                        MACRO testSimpleFile.C
                        MACROARG "\"hsimple409.root\",25000,409,516893,5"
                        FIXTURES_REQUIRED root-io-filemerger-hsimple409-fixture)

      ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                        MACRO testSimpleFile.C
                        MACROARG "\"hsimpleK404.root\",24*25000,404,12047489,30"
                        FIXTURES_REQUIRED root-io-filemerger-hsimpleK404-fixture)

      ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                        MACRO testSimpleFile.C
                        MACROARG "\"hsimple404.root\",25000,404,517275,15"
                        FIXTURES_REQUIRED root-io-filemerger-hsimple404-fixture)

      ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                        MACRO testSimpleFile.C
                        MACROARG "\"hsimple401.root\",25000,401,418016,5"
                        FIXTURES_REQUIRED root-io-filemerger-hsimple401-fixture)

   elseif(${LZ4_VERSION} VERSION_GREATER_EQUAL "1.7.5")
      ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                        MACRO testSimpleFile.C
                        MACROARG "\"hsimpleK404.root\",24*25000,404,12047492,30"
                        FIXTURES_REQUIRED root-io-filemerger-hsimpleK404-fixture)

      ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                        MACRO testSimpleFile.C
                        MACROARG "\"hsimple404.root\",25000,404,517262,15"
                        FIXTURES_REQUIRED root-io-filemerger-hsimple404-fixture)

      if(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
         ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                           MACRO testSimpleFile.C
                           MACROARG "\"hsimple409.root\",25000,409,516975,30"
                           FIXTURES_REQUIRED root-io-filemerger-hsimple409-fixture)
      else()
         ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                           MACRO testSimpleFile.C
                           MACROARG "\"hsimple409.root\",25000,409,517008,15"
                           FIXTURES_REQUIRED root-io-filemerger-hsimple409-fixture)
      endif()
      ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                        MACRO testSimpleFile.C
                        MACROARG "\"hsimple401.root\",25000,401,519976,10"
                        FIXTURES_REQUIRED root-io-filemerger-hsimple401-fixture)
   endif()

   ROOTTEST_ADD_TEST(simplef-default-compr-level9
                     MACRO testSimpleFile.C
                     MACROARG "\"hsimpleF.root\",30*25000,9,12581462,30"
                     FIXTURES_REQUIRED root-io-filemerger-hsimpleF-fixture)
endif()
