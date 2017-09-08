#define TEST_CATEGORY TDataFrameSeq
#include "dataframe_simple_tests.hxx"

#ifdef R__USE_IMT

#include "TROOT.h"
TEST(dataframe_simple, EnableImplicitMT)
{
   ROOT::EnableImplicitMT();
}

#undef TEST_CATEGORY
#define TEST_CATEGORY TDataFrameMT
#include "dataframe_simple_tests.hxx"

#endif
