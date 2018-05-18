#define TEST_CATEGORY RDataFrameSeq
#include "dataframe_regression_tests.hxx"

#ifdef R__USE_IMT

#include "TROOT.h"
TEST(dataframe_regression, EnableImplicitMT)
{
   ROOT::EnableImplicitMT(4);
}

#undef TEST_CATEGORY
#define TEST_CATEGORY RDataFrameMT
#include "dataframe_regression_tests.hxx"

#endif
