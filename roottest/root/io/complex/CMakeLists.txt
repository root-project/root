ROOTTEST_GENERATE_REFLEX_DICTIONARY(classWithComplex
                                    classWithComplex.h
                                    SELECTION classWithComplex_selection.xml
                                    FIXTURES_SETUP root-io-complex-dict-fixture)

ROOTTEST_ADD_TEST(writeClassWithComplex
                  MACRO  execwriteClassWithComplex.C
                  OUTREF execwriteClassWithComplex.ref
                  FIXTURES_REQUIRED root-io-complex-dict-fixture
                  FIXTURES_SETUP root-io-complex-writeClassWithComplex-fixture)

ROOTTEST_ADD_TEST(readClassWithComplex
                  MACRO  execreadClassWithComplex.C
                  OUTREF execreadClassWithComplex.ref
                  FIXTURES_REQUIRED root-io-complex-writeClassWithComplex-fixture)


ROOTTEST_ADD_TEST(writecomplex
                  MACRO  execwritecomplex.C
                  OUTREF execwritecomplex.ref
                  FIXTURES_SETUP root-io-complex-writecomplex-fixture)

ROOTTEST_ADD_TEST(readcomplex
                  COPY_TO_BUILDDIR complexOfilekubuntuROOT5.root  complexOfilekubuntuROOT6.root complexOfileslc6ROOT5.xml complexOfilekubuntuROOT5.xml   complexOfilekubuntuROOT6.xml   complexOfileslc6ROOT5.root
                  MACRO  execreadcomplex.C
                  OUTREF execreadcomplex.ref
                  FIXTURES_REQUIRED root-io-complex-writecomplex-fixture)
