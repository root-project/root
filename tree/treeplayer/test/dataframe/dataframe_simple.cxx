#define TEST_CATEGORY TDataFrameSeq
#define NSLOTS 1U
#include "dataframe_simple_tests.hxx"

#ifdef R__USE_IMT

#undef NSLOTS
#define NSLOTS 4U

#include "TROOT.h"
TEST(dataframe_simple, EnableImplicitMT)
{
   ROOT::EnableImplicitMT(NSLOTS);
}

#undef TEST_CATEGORY
#define TEST_CATEGORY TDataFrameMT
#include "dataframe_simple_tests.hxx"

#endif
