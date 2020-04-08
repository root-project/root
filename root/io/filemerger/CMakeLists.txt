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
      add_custom_target(hsimple-filemergerfile ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root)
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b ${ROOTSYS}/tutorials/hsimple.C -e "{ TFile f(\"hsimple.root\"); TTree *ntuple; f.GetObject(\"ntuple\",ntuple); return ntuple ? 0 : 1; }" > hsimple.log
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${HSimpleDependencies}
                  VERBATIM)
      add_custom_command(
        TARGET  hsimple-filemergerfile
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root
                hsimple.root)
else()
      add_custom_target(hsimple-filemergerfile ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root)
      set(HSimpleDependencies "hsimple")
      add_dependencies(hsimple-filemergerfile ${HSimpleDependencies})
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_BINARY_DIR}/tutorials/hsimple.root
                hsimple.root)
endif()

if(${compression_default} STREQUAL "zlib")
   if(ZLIB_CF)
      ROOTTEST_ADD_TEST(simple-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f9 hsimple9.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9.root\",25000,9,441439,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level1
                  PRECMD ${ROOT_hadd_CMD} -f101 hsimple101.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple101.root\",25000,101,442963,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level6
                  PRECMD ${ROOT_hadd_CMD} -f106 hsimple106.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple106.root\",25000,106,441545,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f109 hsimple109.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109.root\",25000,109,441449,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-default-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f9 hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",25000,9,441449,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f109 hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",25000,109,441456,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simplex2-default-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f9 -a hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",50000,9,870986,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS simple-default-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplex2-zlib-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f109 -a hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",50000,109,870994,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS simple-zlib-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplek-default-compr-deflevel
                  PRECMD ${ROOT_hadd_CMD} -fk hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK.root\",6*25000,209,2297560,30)"
                  DEPENDS hsimple-filemergerfile
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9)

      ROOTTEST_ADD_TEST(simplek-lzma-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -fk209 hsimpleK209.root hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK209.root\",12*25000,209,4583784,30)"
                  DEPENDS hsimple-filemergerfile
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-default-compr-deflevel)

      ROOTTEST_ADD_TEST(simplef-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -ff hsimpleF.root hsimple9.root hsimple209.root hsimple409.root hsimpleK404.root hsimple.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleF.root\",30*25000,9,12889766,30)"
                  DEPENDS hsimple-filemergerfile
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-lz4-compr-level4)

      if(${LZ4_VERSION} VERSION_LESS "1.7.5")
            ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047466,30)"
                           DEPENDS hsimple-filemergerfile
                           DEPENDS simple-lz4-compr-level9
                           DEPENDS simple-lzma-compr-level9
                           DEPENDS simplex2-default-compr-level9
                           DEPENDS simple-default-compr-level9
                           DEPENDS simplek-lzma-compr-level9)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517232,15)"
                           DEPENDS hsimple-filemergerfile)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                           PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,516961,5)"
                           DEPENDS hsimple-filemergerfile)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                           PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,447579,5)"
                           DEPENDS hsimple-filemergerfile)
      elseif(${LZ4_VERSION} VERSION_GREATER_EQUAL "1.7.5")
            ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047466,30)"
                           DEPENDS hsimple-filemergerfile
                           DEPENDS simple-lz4-compr-level9
                           DEPENDS simple-lzma-compr-level9
                           DEPENDS simplex2-default-compr-level9
                           DEPENDS simple-default-compr-level9
                           DEPENDS simplek-lzma-compr-level9)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                           PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517232,15)"
                           DEPENDS hsimple-filemergerfile)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                           PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409, 516961,5)"
                           DEPENDS hsimple-filemergerfile)

            ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                           PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                           COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,447579,5)"
                           DEPENDS hsimple-filemergerfile)
      endif()
   else()
      ROOTTEST_ADD_TEST(simple-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f9 hsimple9.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9.root\",25000,9,430953,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level1
                  PRECMD ${ROOT_hadd_CMD} -f101 hsimple101.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple101.root\",25000,101,414948,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level6
                  PRECMD ${ROOT_hadd_CMD} -f106 hsimple106.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple106.root\",25000,106,431240,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f109 hsimple109.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109.root\",25000,109,430962,25)"
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-default-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f9 hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",25000,9,430961,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simple-zlib-compr-level9-datageneration
                  PRECMD  ${ROOT_hadd_CMD} -f109 hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",25000,109,430970,25)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS hsimple-filemergerfile)

      ROOTTEST_ADD_TEST(simplex2-default-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f9 -a hsimple9x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple9x2.root\",50000,9,850063,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple9x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple92.root
                  DEPENDS simple-default-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplex2-zlib-compr-level9
                  PRECMD  ${ROOT_hadd_CMD} -f109 -a hsimple109x2.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple109x2.root\",50000,109,850071,30)"
                  POSTCMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hsimple109x2.root ${CMAKE_CURRENT_BINARY_DIR}/hsimple1092.root
                  DEPENDS simple-zlib-compr-level9-datageneration)

      ROOTTEST_ADD_TEST(simplek-default-compr-deflevel
                  PRECMD ${ROOT_hadd_CMD} -fk hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK.root\",6*25000,209,2297560,30)"
                  DEPENDS hsimple-filemergerfile
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9)

      ROOTTEST_ADD_TEST(simplek-lzma-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -fk209 hsimpleK209.root hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK209.root\",12*25000,209,4583784,30)"
                  DEPENDS hsimple-filemergerfile
                  DEPENDS filemerger-simple-lz4-compr-level9
                  DEPENDS filemerger-simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-default-compr-deflevel)
      if(${LZ4_VERSION} VERSION_LESS "1.7.5")
         ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                     PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                     COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047489,30)"
                     DEPENDS hsimple-filemergerfile
                     DEPENDS simple-lz4-compr-level9
                     DEPENDS simple-lzma-compr-level9
                     DEPENDS simplex2-default-compr-level9
                     DEPENDS simple-default-compr-level9
                     DEPENDS simplek-lzma-compr-level9)

         ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                        PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517275,15)"
                        DEPENDS hsimple-filemergerfile)
         ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                        PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,516893,5)"
                        DEPENDS hsimple-filemergerfile)

         ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                        PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,418044,5)"
                        DEPENDS hsimple-filemergerfile)

      elseif(${LZ4_VERSION} VERSION_GREATER_EQUAL "1.7.5")
         ROOTTEST_ADD_TEST(simplek-lz4-compr-level4
                     PRECMD ${ROOT_hadd_CMD} -fk404 hsimpleK404.root hsimpleK209.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple92.root
                     COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleK404.root\",24*25000,404,12047466,30)"
                     DEPENDS hsimple-filemergerfile
                     DEPENDS simple-lz4-compr-level9
                     DEPENDS simple-lzma-compr-level9
                     DEPENDS simplex2-default-compr-level9
                     DEPENDS simple-default-compr-level9
                     DEPENDS simplek-lzma-compr-level9)

         ROOTTEST_ADD_TEST(simple-lz4-compr-level4
                        PRECMD ${ROOT_hadd_CMD} -f404 hsimple404.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple404.root\",25000,404,517232,15)"
                        DEPENDS hsimple-filemergerfile)
         if(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                          PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                          COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,516975,5)"
                          DEPENDS hsimple-filemergerfile)
         else()
            ROOTTEST_ADD_TEST(simple-lz4-compr-level9
                          PRECMD ${ROOT_hadd_CMD} -f409 hsimple409.root hsimple.root
                          COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple409.root\",25000,409,516961,5)"
                          DEPENDS hsimple-filemergerfile)
         endif()
         ROOTTEST_ADD_TEST(simple-lz4-compr-level1
                        PRECMD ${ROOT_hadd_CMD} -f401 hsimple401.root hsimple.root
                        COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple401.root\",25000,401,418887,5)"
                        DEPENDS hsimple-filemergerfile)
      endif()

      ROOTTEST_ADD_TEST(simplef-default-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -ff hsimpleF.root hsimple9.root hsimple209.root hsimple409.root hsimpleK404.root hsimple.root hsimple92.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimpleF.root\",30*25000,9,12581392,30)"
                  DEPENDS hsimple-filemergerfile
                  DEPENDS simple-lz4-compr-level9
                  DEPENDS simple-lzma-compr-level9
                  DEPENDS simplex2-default-compr-level9
                  DEPENDS simple-default-compr-level9
                  DEPENDS simplek-lz4-compr-level4)
   endif()
endif()

ROOTTEST_ADD_TEST(simple-lzma-compr-level9
                  PRECMD ${ROOT_hadd_CMD} -f209 hsimple209.root hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b "${CMAKE_CURRENT_SOURCE_DIR}/testSimpleFile.C(\"hsimple209.root\",25000,209,392673,12)"
                  DEPENDS hsimple-filemergerfile)
