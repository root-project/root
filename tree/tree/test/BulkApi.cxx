#include <stdio.h>

#include "Bytes.h"
#include "TBranch.h"
#include "TBufferFile.h"
#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "ROOT/TTreeReaderFast.hxx"
#include "ROOT/TTreeReaderValueFast.hxx"
#include "ROOT/TBulkBranchRead.hxx"


#include "gtest/gtest.h"

class BulkApiTest : public ::testing::Test {
public:
   static constexpr Int_t fEventCount = 1e7;
   const std::string fFileName = "BulkApiTest.root";

protected:
   virtual void SetUp()
   {
      auto hfile = TFile::Open(fFileName.c_str(), "recreate", "TTree float micro benchmark ROOT file");
      hfile->SetCompressionLevel(0); // No compression at all.

      // Otherwise, we keep with the current ROOT defaults.
      auto tree = new TTree("T", "A ROOT tree of floats.");
      float f = 2;
      tree->Branch("myFloat", &f, 320000, 1);
      for (Long64_t ev = 0; ev < fEventCount; ev++) {
         tree->Fill();
         ++f;
      }
      hfile = tree->GetCurrentFile();
      hfile->Write();
      tree->Print();
      printf("Successful write of all events.\n");
      hfile->Close();

      delete hfile;
   }
};

TEST_F(BulkApiTest, stdRead)
{
   auto hfile = TFile::Open(fFileName.c_str());
   printf("Starting read of file %s.\n", fFileName.c_str());
   TStopwatch sw;

   printf("Using standard read APIs.\n");
   // Read via standard APIs.
   TTreeReader myReader("T", hfile);
   TTreeReaderValue<float> myF(myReader, "myFloat");
   Long64_t idx = 0;
   float idx_f = 1;
   Int_t events = fEventCount;
   sw.Start();
   while (myReader.Next()) {
      if (R__unlikely(idx == events)) {
         break;
      }
      idx_f++;
      if (R__unlikely((idx < 16000000) && (abs((*myF) - idx_f) > std::numeric_limits<float>::epsilon()))) {
         printf("Incorrect value on myFloat branch: %f, expected %f (event %lld)\n", *myF, idx_f, idx);
         ASSERT_TRUE(false);
      }
      idx++;
   }
   sw.Stop();
   printf("TTreeReader: Successful read of all events.\n");
   printf("TTreeReader: Total elapsed time (seconds) for standard APIs: %.2f\n", sw.RealTime());
}

TEST_F(BulkApiTest, simpleRead)
{
   auto hfile = TFile::Open(fFileName.c_str());
   printf("Starting read of file %s.\n", fFileName.c_str());
   TStopwatch sw;

   printf("Using inline bulk read APIs.\n");
   TBufferFile branchbuf(TBuffer::kWrite, 32 * 1024);
   TTree *tree = dynamic_cast<TTree *>(hfile->Get("T"));
   ASSERT_TRUE(tree);

   TBranch *branchF = tree->GetBranch("myFloat");
   ASSERT_TRUE(branchF);

   Int_t events = fEventCount;
   float idx_f = 1;
   Long64_t evt_idx = 0;
   while (events) {
      auto count = branchF->GetBulkRead().GetEntriesSerialized(evt_idx, branchbuf);
      ASSERT_GE(count, 0);
      events = events > count ? (events - count) : 0;

      float *entry = reinterpret_cast<float *>(branchbuf.GetCurrent());
      for (Int_t idx = 0; idx < count; idx++) {
         idx_f++;
         Int_t tmp = *reinterpret_cast<Int_t *>(&entry[idx]);
         char *tmp_ptr = reinterpret_cast<char *>(&tmp);
         frombuf(tmp_ptr, entry + idx);

         if (R__unlikely((evt_idx < 16000000) && (entry[idx] != idx_f))) {
            printf("Incorrect value on myFloat branch: %f (event %lld)\n", entry[idx], evt_idx + idx);
            ASSERT_TRUE(false);
         }
      }
      evt_idx += count;
   }
   sw.Stop();
   printf("GetEntriesSerialized: Successful read of all events.\n");
   printf("GetEntriesSerialized: Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
}

TEST_F(BulkApiTest, fastRead)
{
   auto hfile = TFile::Open(fFileName.c_str());
   printf("Starting read of file %s.\n", fFileName.c_str());
   TStopwatch sw;

   printf("Using TTreeReaderFast.\n");
   ROOT::Experimental::TTreeReaderFast myReader("T", hfile);
   ROOT::Experimental::TTreeReaderValueFast<float> myF(myReader, "myFloat");
   myReader.SetEntry(0);
   if (ROOT::Internal::TTreeReaderValueBase::kSetupMatch != myF.GetSetupStatus()) {
      printf("TTreeReaderValueFast<float> failed to initialize.  Status code: %d\n", myF.GetSetupStatus());
      ASSERT_TRUE(false);
   }
   if (myReader.GetEntryStatus() != TTreeReader::kEntryValid) {
      printf("TTreeReaderFast failed to initialize.  Entry status: %d\n", myReader.GetEntryStatus());
      ASSERT_TRUE(false);
   }
   Int_t events = fEventCount;
   Long64_t idx = 0;
   float idx_f = 1;
   for (auto reader_idx : myReader) {
      ASSERT_LT(reader_idx, events);
      ASSERT_EQ(reader_idx, idx);
      idx_f++;
      if (R__unlikely((idx < 16000000) && (abs((*myF) - idx_f) > std::numeric_limits<float>::epsilon()))) {
         printf("Incorrect value on myFloat branch: %f, expected %f (event %lld)\n", *myF, idx_f, idx);
         ASSERT_TRUE(false);
      }
      idx++;
   }
   sw.Stop();
   printf("TTreeReaderFast: Successful read of all events.\n");
   printf("TTreeReaderFast: Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
}
