ROOTTEST_ADD_TESTDIRS()

ROOTTEST_GENERATE_REFLEX_DICTIONARY( classWithComplex classWithComplex.h  SELECTION classWithComplex_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-io-complex-writeClassWithComplex
                  MACRO  execwriteClassWithComplex.C
                  OUTREF execwriteClassWithComplex.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(roottest-root-io-complex-readClassWithComplex
                  MACRO  execreadClassWithComplex.C
                  OUTREF execreadClassWithComplex.ref
                  DEPENDS roottest-root-io-complex-writeClassWithComplex)


ROOTTEST_ADD_TEST(roottest-root-io-complex-writecomplex
                  MACRO  execwritecomplex.C
                  OUTREF execwritecomplex.ref)

ROOTTEST_ADD_TEST(roottest-root-io-complex-readcomplex
                  COPY_TO_BUILDDIR complexOfilekubuntuROOT5.root  complexOfilekubuntuROOT6.root complexOfileslc6ROOT5.xml complexOfilekubuntuROOT5.xml   complexOfilekubuntuROOT6.xml   complexOfileslc6ROOT5.root
                  MACRO  execreadcomplex.C
                  OUTREF execreadcomplex.ref
                  DEPENDS roottest-root-io-complex-writecomplex)
