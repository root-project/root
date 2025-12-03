ROOTTEST_ADD_TEST(assert_tcontext_thread_hoping
                  MACRO assert_tcontext_thread_hoping.cxx
                  OUTREF assert_tcontext_thread_hoping.ref)


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  ROOTTEST_ADD_TEST(FlipOver
                    MACRO execFlipOver.C
                    OUTREF execFlipOver.ref)
endif()

ROOTTEST_ADD_TEST(NestedCollectionStreamerInfo
                  MACRO execNestedCollectionStreamerInfo.cxx+
                  OUTREF execNestedCollectionStreamerInfo.ref)

if(geom)
  ROOTTEST_ADD_TEST(RelativePath
                    MACRO execRelativePath.C
                    COPY_TO_BUILDDIR geodemo.root
                    OUTREF execRelativePath.ref)
endif()

ROOTTEST_ADD_TEST(StreamerInfoHash
                  MACRO execStreamerInfoHash.cxx+
                  OUTREF execStreamerInfoHash.ref)

if(geom)
  ROOTTEST_ADD_TEST(StreamerInfoList
                    MACRO execStreamerInfoList.C
                    OUTREF execStreamerInfoList.ref
                    COPY_TO_BUILDDIR geodemo.root)
endif(geom)
