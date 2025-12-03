ROOTTEST_ADD_TEST(ROOT-7249
                  MACRO AccidentalOwnership.C)


ROOTTEST_ADD_TEST(HistArray
                  MACRO execHistArray.cxx+
                  COPY_TO_BUILDDIR HistArray.root
                  OUTREF execHistArray.ref)

ROOTTEST_ADD_TEST(RangeExpression
                  MACRO execRangeExpression.C
                  OUTREF execRangeExpression.ref)

ROOTTEST_ADD_TEST(TClonesArrayAbsorb
                  MACRO execTClonesArrayAbsorb.C
                  OUTREF execTClonesArrayAbsorb.ref)

ROOTTEST_ADD_TEST(CATTreeClear
                  MACRO runCATTreeClear.C
                  OUTREF CATTreeClear.ref)

ROOTTEST_ADD_TEST(TMap
                  MACRO runTMap.C
                  OUTREF TMap.ref)

ROOTTEST_ADD_TEST(tbits
                  MACRO runtbits.C
                  OUTREF tbits.ref)

ROOTTEST_ADD_TEST(Missing
                  MACRO execMissing.C
                  OUTREF execMissing.oref
                  ERRREF execMissing.eref )

if (32BIT OR MSVC)
  ROOTTEST_ADD_TEST(TExMap
                    MACRO runTExMap.C
                    OUTREF runTExMap32.oref
                    ERRREF runTExMap32.eref )
else()
  ROOTTEST_ADD_TEST(TExMap
                    MACRO runTExMap.C
                    OUTREF runTExMap.oref
                    ERRREF runTExMap.eref )
endif()

ROOTTEST_ADD_TEST(DeleteWarning
                  MACRO runDeleteWarning.C
                  OUTCNVCMD sed -e s:0x[0-9a-fA-F]*:0xRemoved:g
                  OUTREF DeleteWarning.ref)
