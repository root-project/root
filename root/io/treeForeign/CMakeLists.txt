
if(${compression_default} STREQUAL "lz4")
          ROOTTEST_ADD_TEST(testForeignDraw
                  MACRO Run.C
                  PRECMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/def.C+
                  OUTREF testForeignDrawLZ4.ref
                  DEPENDS test-dataset)
else(${compression_default} STREQUAL "zlib")
   if(ZLIB_CF)
          ROOTTEST_ADD_TEST(testForeignDraw
                  MACRO Run.C
                  PRECMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/def.C+
                  OUTREF testForeignDrawZLIB_builtinzlib.ref
                  DEPENDS test-dataset)
   else()
	  ROOTTEST_ADD_TEST(testForeignDraw
                  MACRO Run.C
                  PRECMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/def.C+
                  OUTREF testForeignDrawZLIB.ref
                  DEPENDS test-dataset)
   endif()
endif()
