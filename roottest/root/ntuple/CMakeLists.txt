ROOTTEST_ADD_TEST(test_Basics
                  MACRO basics.C
                  OUTREF test_basics.ref)

# issue 13910: MakeField() must fail cleanly for invalid types and keep the
# model/interpreter usable.
ROOTTEST_ADD_TEST(test_MakeFieldInvalid
                  MACRO makefield_invalid.C
                  PASSREGEX "all MakeField failure checks passed")

ROOTTEST_ADD_TESTDIRS()
