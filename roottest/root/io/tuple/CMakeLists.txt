if(NOT MSVC)
  ROOTTEST_ADD_TEST(Tuple
                    MACRO execTuple.cxx+
                    COPY_TO_BUILDDIR tuple.macos.root tuple.x86_64.root
                    OUTREF execTuple.ref)
endif()
