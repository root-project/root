ROOTTEST_ADD_TEST(execKeyOrder
	          MACRO execKeyOrder.C
		  OUTREF execKeyOrder.ref)


ROOTTEST_ADD_TEST(execCreateAndMerge
                  MACRO execCreateAndMerge.C
                  OUTREF references/execCreateAndMerge.ref)

ROOTTEST_ADD_TEST(execMergeMulti
                  MACRO execMergeMulti.C
                  OUTREF references/execMergeMulti.ref)

ROOTTEST_ADD_TEST(execFileMerger
                  MACRO execFileMerger.C
                  OUTREF references/execFileMerger.ref)

# FIXME: Should be CTEST fixtures
ROOTTEST_ADD_TEST(datagen-hadd-mfile12
                     COMMAND ${ROOT_hadd_CMD} -f mfile1-2.root mfile1.root mfile2.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mfile34
                     COMMAND ${ROOT_hadd_CMD} -f mfile3-4.root mfile3.root mfile4.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mzfile12
                     COMMAND ${ROOT_hadd_CMD} -f209 mzfile1-2.root mfile1.root mfile2.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mzfile34
                     COMMAND ${ROOT_hadd_CMD} -f209 mzfile3-4.root mfile3.root mfile4.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mzlibfile12
                     COMMAND ${ROOT_hadd_CMD} -f101 mzlibfile1-2.root mfile1.root mfile2.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mzlibfile34
                     COMMAND ${ROOT_hadd_CMD} -f101 mzlibfile3-4.root mfile3.root mfile4.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mlz4file12
                     COMMAND ${ROOT_hadd_CMD} -f404 mlz4file1-2.root mfile1.root mfile2.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mlz4file34
                     COMMAND ${ROOT_hadd_CMD} -f404 mlz4file3-4.root mfile3.root mfile4.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mzstdfile12
                     COMMAND ${ROOT_hadd_CMD} -f505 mzstdfile1-2.root mfile1.root mfile2.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

ROOTTEST_ADD_TEST(datagen-hadd-mzstdfile34
                     COMMAND ${ROOT_hadd_CMD} -f505 mzstdfile3-4.root mfile3.root mfile4.root
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS execMergeMulti)

if(ZLIB_CF)
  ROOTTEST_ADD_TEST(merged-zlib
                    PRECMD  ${ROOT_hadd_CMD} -f101 mzlibfile1-4.root mzlibfile1-2.root mzlibfile3-4.root
                    COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mzlibfile1-4.root\",101,4804,20)"
                    DEPENDS datagen-hadd-mzlibfile12
                    DEPENDS datagen-hadd-mzlibfile34)
else()
  ROOTTEST_ADD_TEST(merged-zlib
                    PRECMD  ${ROOT_hadd_CMD} -f101 mzlibfile1-4.root mzlibfile1-2.root mzlibfile3-4.root
                    COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mzlibfile1-4.root\",101,5014,20)"
                    DEPENDS datagen-hadd-mzlibfile12
                    DEPENDS datagen-hadd-mzlibfile34)
endif()

ROOTTEST_ADD_TEST(merged-lzma
                  PRECMD ${ROOT_hadd_CMD} -f207 mzfile1-4.root mzfile1-2.root mzfile3-4.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mzfile1-4.root\",207,4455,20)"
                  DEPENDS datagen-hadd-mzfile12
                  DEPENDS datagen-hadd-mzfile34)

if(${LZ4_VERSION} VERSION_LESS "1.7.5")
   ROOTTEST_ADD_TEST(merged-lz4
                  PRECMD ${ROOT_hadd_CMD} -f404 mlz4file1-4.root mlz4file1-2.root mlz4file3-4.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mlz4file1-4.root\",404,5441,5)"
                  DEPENDS datagen-hadd-mlz4file12
                  DEPENDS datagen-hadd-mlz4file34)
elseif(${LZ4_VERSION} VERSION_GREATER_EQUAL "1.7.5")
   ROOTTEST_ADD_TEST(merged-lz4
                  PRECMD ${ROOT_hadd_CMD} -f404 mlz4file1-4.root mlz4file1-2.root mlz4file3-4.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mlz4file1-4.root\",404,5407,5)"
                  DEPENDS datagen-hadd-mlz4file12
                  DEPENDS datagen-hadd-mlz4file34)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
   if(${ZSTD_VERSION} VERSION_LESS_EQUAL "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
            PRECMD ${ROOT_hadd_CMD} -f505 mzstdfile1-4.root mzstdfile1-2.root mzstdfile3-4.root
            COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mzstdfile1-4.root\",505,4533,20)"
            DEPENDS datagen-hadd-mzstdfile12
            DEPENDS datagen-hadd-mzstdfile34)
   else(${ZSTD_VERSION} VERSION_GREATER "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
            PRECMD ${ROOT_hadd_CMD} -f505 mzstdfile1-4.root mzstdfile1-2.root mzstdfile3-4.root
            COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mzstdfile1-4.root\",505,4456,20)"
            DEPENDS datagen-hadd-mzstdfile12
            DEPENDS datagen-hadd-mzstdfile34)
   endif()
else()
   if(${ZSTD_VERSION} VERSION_LESS_EQUAL "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
                  PRECMD ${ROOT_hadd_CMD} -f505 mzstdfile1-4.root mzstdfile1-2.root mzstdfile3-4.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mzstdfile1-4.root\",505,4529,20)"
                  DEPENDS datagen-hadd-mzstdfile12
                  DEPENDS datagen-hadd-mzstdfile34)
   else(${ZSTD_VERSION} VERSION_GREATER "1.3.3")
      ROOTTEST_ADD_TEST(merged-zstd
                  PRECMD ${ROOT_hadd_CMD} -f505 mzstdfile1-4.root mzstdfile1-2.root mzstdfile3-4.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testMergedFile.C(\"mzstdfile1-4.root\",505,4454,20)"
                  DEPENDS datagen-hadd-mzstdfile12
                  DEPENDS datagen-hadd-mzstdfile34)
   endif()
endif()

#TBD: add file generation as test fixture
if(NOT TARGET hsimple)
      add_test(NAME roottest-root-io-filemerger-hsimple-create
               COMMAND ${ROOT_root_CMD} -q -l -b ${ROOTSYS}/tutorials/hsimple.C -e "{ TFile f(\"hsimple.root\"); TTree *ntuple; f.GetObject(\"ntuple\",ntuple); return ntuple ? 0 : 1; }"
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      add_test(NAME roottest-root-io-filemerger-hsimple
               COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root hsimple.root)
      set_property(TEST roottest-root-io-filemerger-hsimple PROPERTY DEPENDS roottest-root-io-filemerger-hsimple-create)
else()
      add_test(NAME roottest-root-io-filemerger-hsimple
               COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/tutorials/hsimple.root hsimple.root)
endif()

if(${compression_default} STREQUAL "zlib")
   if(ZLIB_CF)
      ROOTTEST_ADD_TEST(simple-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f9 hsimple9.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9.root\",25000,9,441539,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level1
                  PRECMD ${ROOT_hadd_CMD} -f101 hsimple101.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple101.root\",25000,101,443064,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level6
                  PRECMD ${ROOT_hadd_CMD} -f106 hsimple106.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple106.root\",25000,106,441650,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f109 hsimple109.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109.root\",25000,109,441549,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-default-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f9 hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",25000,9,441549,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f109 hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",25000,109,441556,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simplex2-default-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f9 -a hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",50000,9,871086,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS simple-default-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplex2-zlib-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f109 -a hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",50000,109,871092,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS simple-zlib-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplek-default-compr-deflevel
                  PRECMD ${ROOT_hadd_CMD} -fk hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK.root\",6*25000,209,2297660,30)"
                  DEPENDS roottest-root-io-filemerger-hsimple
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9)

      ROOTTEST_ADD_TEST(simplek-lzma-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -fk209 hsimpleK209.root hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK209.root\",12*25000,209,4583884,30)"
                  DEPENDS roottest-root-io-filemerger-hsimple
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-default-compr-deflevel)

      ROOTTEST_ADD_TEST(simplef-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -ff hsimpleF.root hsimple9.root hsimple209.root hsimple409.root hsimpleK404.root hsimple.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleF.root\",30*25000,9,12889870,30)"
                  DEPENDS roottest-root-io-filemerger-hsimple
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-lz4-compr-level4)

      if(${LZ4_VERSION} VERSION_LESS "1.7.5")
            ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047589,30)"
                           DEPENDS roottest-root-io-filemerger-hsimple
                           DEPENDS simple-lz4-compr-level9
                           DEPENDS simple-lzma-compr-level9
                           DEPENDS simplex2-default-compr-level9
                           DEPENDS simple-default-compr-level9
                           DEPENDS simplek-lzma-compr-level9)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517355,15)"
                           DEPENDS roottest-root-io-filemerger-hsimple)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                           PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,517081,5)"
                           DEPENDS roottest-root-io-filemerger-hsimple)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                           PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,447725,5)"
                           DEPENDS roottest-root-io-filemerger-hsimple)
      elseif(${LZ4_VERSION} VERSION_GREATER_EQUAL "1.7.5")
            ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047589,30)"
                           DEPENDS roottest-root-io-filemerger-hsimple
                           DEPENDS simple-lz4-compr-level9
                           DEPENDS simple-lzma-compr-level9
                           DEPENDS simplex2-default-compr-level9
                           DEPENDS simple-default-compr-level9
                           DEPENDS simplek-lzma-compr-level9)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517355,15)"
                           DEPENDS roottest-root-io-filemerger-hsimple)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                           PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409, 517081,5)"
                           DEPENDS roottest-root-io-filemerger-hsimple)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                           PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,447725,5)"
                           DEPENDS roottest-root-io-filemerger-hsimple)
      endif()
   else()
      ROOTTEST_ADD_TEST(simple-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f9 hsimple9.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9.root\",25000,9,431056,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level1
                  PRECMD ${ROOT_hadd_CMD} -f101 hsimple101.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple101.root\",25000,101,415079,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level6
                  PRECMD ${ROOT_hadd_CMD} -f106 hsimple106.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple106.root\",25000,106,431347,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f109 hsimple109.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109.root\",25000,109,431067,25)"
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-default-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f9 hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",25000,9,431067,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f109 hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",25000,109,431073,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS roottest-root-io-filemerger-hsimple)

      ROOTTEST_ADD_TEST(simplex2-default-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f9 -a hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",50000,9,850169,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS simple-default-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplex2-zlib-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f109 -a hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",50000,109,850173,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS simple-zlib-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplek-default-compr-deflevel
                  PRECMD ${ROOT_hadd_CMD} -fk hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK.root\",6*25000,209,2297660,30)"
                  DEPENDS roottest-root-io-filemerger-hsimple
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9)

      ROOTTEST_ADD_TEST(simplek-lzma-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -fk209 hsimpleK209.root hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK209.root\",12*25000,209,4583884,30)"
                  DEPENDS roottest-root-io-filemerger-hsimple
                  DEPENDS filemerger-simple-lz4-compr-level9
                  DEPENDS filemerger-simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-default-compr-deflevel)
      if(${LZ4_VERSION} VERSION_LESS "1.7.5")
         ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                     PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                     COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047489,30)"
                     DEPENDS roottest-root-io-filemerger-hsimple
                     DEPENDS simple-lz4-compr-level9
                     DEPENDS simple-lzma-compr-level9
                     DEPENDS simplex2-default-compr-level9
                     DEPENDS simple-default-compr-level9
                     DEPENDS simplek-lzma-compr-level9)

         ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                        PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517275,15)"
                        DEPENDS roottest-root-io-filemerger-hsimple)
         ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                        PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,516893,5)"
                        DEPENDS roottest-root-io-filemerger-hsimple)

         ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                        PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,418044,5)"
                        DEPENDS roottest-root-io-filemerger-hsimple)

      elseif(${LZ4_VERSION} VERSION_GREATER_EQUAL "1.7.5")
         ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                     PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                     COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047589,30)"
                     DEPENDS roottest-root-io-filemerger-hsimple
                     DEPENDS simple-lz4-compr-level9
                     DEPENDS simple-lzma-compr-level9
                     DEPENDS simplex2-default-compr-level9
                     DEPENDS simple-default-compr-level9
                     DEPENDS simplek-lzma-compr-level9)

         ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                        PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517355,15)"
                        DEPENDS roottest-root-io-filemerger-hsimple)
         if(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                          PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                          COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,516975,5)"
                          DEPENDS roottest-root-io-filemerger-hsimple)
         else()
            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                          PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                          COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,517081,5)"
                          DEPENDS roottest-root-io-filemerger-hsimple)
         endif()
         ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                        PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,419026,14)"
                        DEPENDS roottest-root-io-filemerger-hsimple)
      endif()

      ROOTTEST_ADD_TEST(simplef-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -ff hsimpleF.root hsimple9.root hsimple209.root hsimple409.root hsimpleK404.root hsimple.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleF.root\",30*25000,9,12581494,30)"
                  DEPENDS roottest-root-io-filemerger-hsimple
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-lz4-compr-level4)
   endif()
endif()

ROOTTEST_ADD_TEST(simple-lzma-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f209 hsimple209.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple209.root\",25000,209,392777,12)"
                  DEPENDS roottest-root-io-filemerger-hsimple)
