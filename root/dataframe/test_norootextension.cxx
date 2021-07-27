#include <ROOT/RDataFrame.hxx>
#include <ROOTUnitTestSupport.h>
#include <TROOT.h>
#include <TSystem.h>

#include <gtest/gtest.h>

#include <thread>

struct NoROOTExtension : public ::testing::TestWithParam<bool> {
   const std::string fname = "rdf_test_norootextension";

   NoROOTExtension()
   {
      if (GetParam())
         ROOT::EnableImplicitMT(std::min(4u, std::thread::hardware_concurrency()));
      ROOT::RDataFrame(10).Define("x", [] { return 42; }).Snapshot<int>("t", fname, {"x"});
   }

   ~NoROOTExtension()
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
      gSystem->Unlink(fname.c_str());
   }
};

TEST_P(NoROOTExtension, Read)
{
   auto m = ROOT::RDataFrame("t", fname).Max<int>("x");
   ASSERT_NO_THROW(m.GetValue());
   EXPECT_EQ(*m, 42);
}

INSTANTIATE_TEST_SUITE_P(Seq, NoROOTExtension, ::testing::Values(false));

#ifdef R__USE_IMT
   INSTANTIATE_TEST_SUITE_P(MT, NoROOTExtension, ::testing::Values(true));
#endif
