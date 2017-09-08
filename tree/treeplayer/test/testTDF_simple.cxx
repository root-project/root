#define TEST_CATEGORY TDataFrameSeq
#include "testTDF_simple_tests.hxx"

#ifdef R__USE_IMT

#include "TROOT.h"
TEST(testTDF_simple, EnableImplicitMT)
{
   ROOT::EnableImplicitMT();
}

#undef TEST_CATEGORY
#define TEST_CATEGORY TDataFrameMT
#include "testTDF_simple_tests.hxx"

#endif
