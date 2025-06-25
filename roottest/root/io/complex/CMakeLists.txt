ROOTTEST_GENERATE_REFLEX_DICTIONARY( classWithComplex classWithComplex.h  SELECTION classWithComplex_selection.xml)

ROOTTEST_ADD_TEST(writeClassWithComplex
                  MACRO  execwriteClassWithComplex.C
                  OUTREF execwriteClassWithComplex.ref
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(readClassWithComplex
                  MACRO  execreadClassWithComplex.C
                  OUTREF execreadClassWithComplex.ref
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS writeClassWithComplex)


ROOTTEST_ADD_TEST(writecomplex
                  MACRO  execwritecomplex.C
                  ${WILLFAIL_ON_WIN32}
                  OUTREF execwritecomplex.ref)

ROOTTEST_ADD_TEST(readcomplex
                  COPY_TO_BUILDDIR complexOfilekubuntuROOT5.root  complexOfilekubuntuROOT6.root complexOfileslc6ROOT5.xml complexOfilekubuntuROOT5.xml   complexOfilekubuntuROOT6.xml   complexOfileslc6ROOT5.root
                  MACRO  execreadcomplex.C
                  OUTREF execreadcomplex.ref
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS writecomplex)
