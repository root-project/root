ROOTTEST_ADD_TEST(loadLHCb
                  MACRO loadLHCb.C
                  LABELS longtest io
                  FIXTURES_SETUP root-io-evolution-versions-lhcb-load)

ROOTTEST_ADD_TEST(runLHCb
                  MACRO runLHCb.C
                  OUTREF LHCb.ref
                  COPY_TO_BUILDDIR lhcb.root
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-versions-lhcb-load
                  OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=na,g)
