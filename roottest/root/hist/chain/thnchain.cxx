#include "gtest/gtest.h"

#include <TFile.h>
#include <TH1.h>
#include <THnChain.h>
#include <THnSparse.h>
#include <TString.h>
#include <TSystem.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

struct TestFile : public TFile {
   TestFile(std::string file_name) :
      TFile(file_name.c_str(), "RECREATE"), fFile_name(file_name) {}

   ~TestFile()
   {
      gSystem->Unlink(fFile_name.c_str());
   }

   const std::string fFile_name;
};

class THnChainTest : public ::testing::Test {
 public:
   THnChainTest()
   {
      // Create histograms spread over a number of files.
      for (int i = 0; i < (int)files.size(); ++i) {
         TString file_name;
         file_name.Form("f%d.root", i);

         files[i].reset(new TestFile(file_name.Data()));

         Int_t bins[1] = {10};
         Double_t xmin[1] = {0};
         Double_t xmax[1] = {1};
         THnSparseF h("h", "h", 1, bins, xmin, xmax);

         Double_t data[] = {0.55}; // Fills bin 6.
         h.Fill(data);

         h.Write();
      }

      // Close files here so they can be reopened in test, but
      // keep the `TestFile`s alive so they are not removed.
      for (const auto& file : files) {
         file->Close();
      }

      // Setup the chain with the existing histograms.
      chain.reset(new THnChain("h"));

      for (const auto& file : files) {
         chain->AddFile(file->GetName());
      }
   }

   std::array<std::unique_ptr<TestFile>, 10> files;

   std::unique_ptr<THnChain> chain;
};

// Obtain the full projection and verify
// it has the expected number of entries.
TEST_F(THnChainTest, SimpleProjection)
{
   const auto projection = chain->Projection(0);
   ASSERT_TRUE(projection);

   EXPECT_EQ(files.size(), projection->GetBinContent(6));
}

// Project an empty sub-range and verify it has no entries in projection.
// FIXME: Overflow bins should still accurately reflect the original number of entries
TEST_F(THnChainTest, EmptySubRange)
{
   TAxis* axis = chain->GetAxis(0);
   axis->SetRange(0, 1);

   const auto projection = chain->Projection(0);
   ASSERT_TRUE(projection);

   EXPECT_EQ(1, projection->GetNbinsX());
   EXPECT_EQ(0, projection->GetBinContent(1));

   // FIXME: We should still be able to recover the original
   // number of entries from overflow/underflow bins.
   // EXPECT_EQ(0, projection->GetBinContent(0));
   // EXPECT_EQ(files.size(), projection->GetBinContent(2));
}
