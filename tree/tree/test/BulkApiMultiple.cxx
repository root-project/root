#include <stdio.h>

#include "TBranch.h"
#include "TBufferFile.h"
#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "ROOT/TTreeReaderFast.hxx"
#include "ROOT/TTreeReaderValueFast.hxx"

#include "gtest/gtest.h"

class BulkApiMultipleTest : public ::testing::Test {
public:
   static constexpr Int_t fEventCount = 1e7;
   const std::string fFileName = "BulkApiMultipleTest.root";

protected:
   virtual void SetUp()
   {
      auto hfile = new TFile(fFileName.c_str(), "RECREATE", "TTree float and double micro benchmark ROOT file");
      hfile->SetCompressionLevel(0); // No compression at all.

      // Otherwise, we keep with the current ROOT defaults.
      auto tree = new TTree("T", "A ROOT tree of floats.");
      float f = 2;
      double g = 3;
      TBranch *branch2 = tree->Branch("myFloat", &f, 320000, 1);
      TBranch *branch3 = tree->Branch("myDouble", &g, 320000, 1);
      branch2->SetAutoDelete(kFALSE);
      branch3->SetAutoDelete(kFALSE);
      for (Long64_t ev = 0; ev < fEventCount; ev++) {
         tree->Fill();
         f ++;
         g ++;
      }
      hfile = tree->GetCurrentFile();
      hfile->Write();
      tree->Print();
      printf("Successful write of all events.\n");
      hfile->Close();

      delete hfile;
   }
};

TEST_F(BulkApiMultipleTest, stdRead)
{
   auto hfile = TFile::Open(fFileName.c_str());
   printf("Starting read of file %s.\n", fFileName.c_str());
   TStopwatch sw;

   printf("Using standard read APIs.\n");
   // Read via standard APIs.
   TTreeReader myReader("T", hfile);
   TTreeReaderValue<float> myF(myReader, "myFloat");
   TTreeReaderValue<double> myG(myReader, "myDouble");
   Long64_t idx = 0;
   float idx_f = 1;
   double idx_g = 2;
   Int_t events = fEventCount;
   sw.Start();
   while (myReader.Next()) {
      if (R__unlikely(idx == events)) {break;}
      idx_f++;
      idx_g++;
      if (R__unlikely((idx < 16000000) && (*myF != idx_f))) {
         printf("Incorrect value on myFloat branch: %f, expected %f (event %lld)\n", *myF, idx_f, idx);
         ASSERT_TRUE(false);
      }
      if (R__unlikely((idx < 15000000) && (*myG != idx_g))) {
         printf("Incorrect value on myDouble branch: %f, expected %f (event %lld)\n", *myG, idx_g, idx);
         ASSERT_TRUE(false);
      }
      idx++;
   }
   sw.Stop();
   printf("TTreeReader: Successful read of all events.\n");
   printf("TTreeReader: Total elapsed time (seconds) for standard APIs: %.2f\n", sw.RealTime());
}

TEST_F(BulkApiMultipleTest, fastRead)
{
   auto hfile = TFile::Open(fFileName.c_str());
   printf("Starting read of file %s.\n", fFileName.c_str());
   TStopwatch sw;

   printf("Using TTreeReaderFast.\n");
   ROOT::Experimental::TTreeReaderFast myReader("T", hfile);
   ROOT::Experimental::TTreeReaderValueFast<float> myF(myReader, "myFloat");
   ROOT::Experimental::TTreeReaderValueFast<double> myG(myReader, "myDouble");
   myReader.SetEntry(0);
   if (ROOT::Internal::TTreeReaderValueBase::kSetupMatch != myF.GetSetupStatus()) {
      printf("TTreeReaderValueFast<float> failed to initialize.  Status code: %d\n", myF.GetSetupStatus());
      ASSERT_TRUE(false);
   }
   if (ROOT::Internal::TTreeReaderValueBase::kSetupMatch != myG.GetSetupStatus()) {
      printf("TTreeReaderValueFast<double> failed to initialize.  Status code: %d\n", myG.GetSetupStatus());
      ASSERT_TRUE(false);
   }
   if (myReader.GetEntryStatus() != TTreeReader::kEntryValid) {
      printf("TTreeReaderFast failed to initialize.  Entry status: %d\n", myReader.GetEntryStatus());
      ASSERT_TRUE(false);
   }
   Int_t events = fEventCount;
   Long64_t idx = 0;
   float idx_f = 1;
   double idx_g = 2;
   for (auto reader_idx : myReader) {
      ASSERT_LT(reader_idx, events);
      ASSERT_EQ(reader_idx, idx);
      idx_f++;
      idx_g++;
      if (R__unlikely((idx < 16000000) && (*myF != idx_f))) {
         printf("Incorrect value on myFloat branch: %f, expected %f (event %lld)\n", *myF, idx_f, idx);
         ASSERT_TRUE(false);
      }
      if (R__unlikely((idx < 15000000) && (*myG != idx_g))) {
         printf("Incorrect value on myDouble branch: %f, expected %f (event %lld)\n", *myG, idx_g, idx);
         ASSERT_TRUE(false);
      }
      idx++;
   }
   sw.Stop();
   printf("TTreeReaderFast: Successful read of all events.\n");
   printf("TTreeReaderFast: Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
}
