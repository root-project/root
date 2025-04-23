ROOTTEST_GENERATE_REFLEX_DICTIONARY(paramPackDict OPTIONS --fail_on_warnings SELECTION paramPackSel.xml paramPack.h)

ROOTTEST_ADD_TEST(execParamPack
                  MACRO execParamPack.C
                  COPY_TO_BUILDDIR another.rootmap
                  DEPENDS paramPackDict-libgen-build
                  LABELS roottest regression cling)
