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

static const Long64_t gRollOver = std::pow(10, (std::numeric_limits<float>::digits10-1));

class BulkApiMultipleTest : public ::testing::Test {
public:
   static constexpr Long64_t fEventCount = 1e7;
   const std::string fFileName = "BulkApiMultipleTest.root";
   static constexpr float fFloatInitial = 2.0;
   static constexpr double fDoubleInitial = 3.0;

protected:
   virtual void SetUp()
   {
      auto hfile = new TFile(fFileName.c_str(), "RECREATE", "TTree float and double micro benchmark ROOT file");
      hfile->SetCompressionLevel(0); // No compression at all.

      // Otherwise, we keep with the current ROOT defaults.
      auto tree = new TTree("T", "A ROOT tree of floats.");

      float f = fFloatInitial;
      double g = fDoubleInitial;
      bool b = true;
      short s  = 4;
      int i = 5;
      Long64_t ll = (1<<20) + 1;
      char c = 7;

      TBranch *branch2 = tree->Branch("myFloat", &f, 320000, 1);
      TBranch *branch3 = tree->Branch("myDouble", &g, 320000, 1);
      TBranch *branch4 = tree->Branch("myBool", &b, 320000, 1);
      TBranch *branch5 = tree->Branch("myShort", &s, 320000, 1);
      TBranch *branch6 = tree->Branch("myInt", &i, 320000, 1);
      TBranch *branch7 = tree->Branch("myLongLong", &ll, 320000, 1);
      TBranch *branch8 = tree->Branch("myChar", &c, 320000, 1);

      branch2->SetAutoDelete(kFALSE);
      branch3->SetAutoDelete(kFALSE);
      branch4->SetAutoDelete(kFALSE);
      branch5->SetAutoDelete(kFALSE);
      branch6->SetAutoDelete(kFALSE);
      branch7->SetAutoDelete(kFALSE);
      branch8->SetAutoDelete(kFALSE);

      for (Long64_t ev = 0; ev < fEventCount; ev++) {
         tree->Fill();
         if (ev && (ev % gRollOver) == 0) {
            f = 1.0;
            g = 2.0;
         }
         f ++;
         g ++;
         b = !b;
         s ++;
         i ++;
         ll ++;
         c ++;
      }
      hfile = tree->GetCurrentFile();
      hfile->Write();
      //tree->Print();
      printf("Successful write of all events.\n");
      hfile->Close();

      delete hfile;
   }
};


template <typename T>
bool rollover(Long64_t /* ev */, T & /* value */)
{
   return false;
}

bool rollover(Long64_t ev, float &value)
{
   if (ev && (ev % gRollOver) == 0) {
      value = BulkApiMultipleTest::fFloatInitial;
      return true;
   }
   return false;
}

bool rollover(Long64_t ev, double &value)
{
   if (ev && (ev % gRollOver) == 0) {
      value = BulkApiMultipleTest::fDoubleInitial;
      return true;
   }
   return false;
}

template <typename T>
void increment(Long64_t ev, T &value)
{
   if (!rollover(ev, value))
      ++value;
}

void increment(Long64_t /* ev */, bool &value)
{
   value = !value;
}

template <typename T>
bool compare(T value, T expected)
{
   return (ULong64_t)value == (ULong64_t)expected;
}

template <typename T>
void SimpleBulkReadFunc(const char *filename, const char *treename, const char *branchname, T initialvalue)
{
   auto hfile = TFile::Open(filename);
   printf("Starting read of file %s.\n", filename);
   TStopwatch sw;

   printf("Using outlined bulk read APIs.\n");
   TBufferFile branchbuf(TBuffer::kWrite, 32 * 1024);
   auto tree = hfile->Get<TTree>(treename);
   ASSERT_TRUE(tree);
   EXPECT_EQ(tree->GetEntries(), BulkApiMultipleTest::fEventCount);

   TBranch *branch = tree->GetBranch(branchname);
   ASSERT_TRUE(branch);
   //branch->GetListOfBaskets()->ls();

   T value = initialvalue;
   Long64_t evt_idx = 0;
   while (auto count = branch->GetBulkRead().GetBulkEntries(evt_idx, branchbuf)) {
      if (count < 0)
         break;
      auto entry = reinterpret_cast<T *>(branchbuf.GetCurrent());
      for (Int_t idx = 0; idx < count; idx++) {
         EXPECT_TRUE(compare(entry[idx], value))
            << "In tree " << treename << " Incorrect value for " << branchname
            << " branch: " << entry[idx] << ", expected " << value << " at entry #" << evt_idx + idx
            << (tree->Scan(branchname, "", "", 10, evt_idx + idx - 3), '\n');
         increment( evt_idx + idx, value );
      }
      evt_idx += count;
   }
   sw.Stop();
   EXPECT_EQ(evt_idx, BulkApiMultipleTest::fEventCount);
   printf("GetBulkEntries: Successful read of all events in %s.\n", treename);
   printf("GetBulkEntries: Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
   delete hfile;
}

TEST_F(BulkApiMultipleTest, simpleReadF)
{
   SimpleBulkReadFunc<float>(fFileName.c_str(), "T", "myFloat", 2);
}

TEST_F(BulkApiMultipleTest, simpleReadD)
{
   SimpleBulkReadFunc<double>(fFileName.c_str(), "T", "myDouble", 3);
}

TEST_F(BulkApiMultipleTest, simpleReadB)
{
   SimpleBulkReadFunc<bool>(fFileName.c_str(), "T", "myBool", true);
}

TEST_F(BulkApiMultipleTest, simpleReadS)
{
   SimpleBulkReadFunc<short>(fFileName.c_str(), "T", "myShort", 4);
}

TEST_F(BulkApiMultipleTest, simpleReadI)
{
   SimpleBulkReadFunc<int>(fFileName.c_str(), "T", "myInt", 5);
}

TEST_F(BulkApiMultipleTest, simpleReadLL)
{
   SimpleBulkReadFunc<Long64_t>(fFileName.c_str(), "T", "myLongLong", (1<<20) + 1);
}

TEST_F(BulkApiMultipleTest, simpleReadC)
{
   SimpleBulkReadFunc<char>(fFileName.c_str(), "T", "myChar", 7);
}

template <typename T>
void SimpleSerializedReadFunc(const char *filename, const char *treename, const char *branchname, T initialvalue)
{
   auto hfile = TFile::Open(filename);
   printf("Starting read of file %s.\n", filename);
   TStopwatch sw;

   printf("Using outlined bulk read APIs.\n");
   TBufferFile branchbuf(TBuffer::kWrite, 32 * 1024);
   auto tree = hfile->Get<TTree>(treename);
   ASSERT_TRUE(tree);
   EXPECT_EQ(tree->GetEntries(), BulkApiMultipleTest::fEventCount);

   TBranch *branch = tree->GetBranch(branchname);
   ASSERT_TRUE(branch);
   //branch->GetListOfBaskets()->ls();

   T value = initialvalue;
   Long64_t evt_idx = 0;
   while (auto count = branch->GetBulkRead().GetEntriesSerialized(evt_idx, branchbuf)) {
      if (count < 0)
         break;

      auto entry = reinterpret_cast<T *>(branchbuf.GetCurrent());
      for (Int_t idx = 0; idx < count; idx++) {
         auto tmp = *reinterpret_cast<T *>(&entry[idx]);
         auto tmp_ptr = reinterpret_cast<char *>(&tmp);
           // std::cerr << "BEFORE In tree " << treename << " Incorrect value for " << branchname
            //          << " branch: " << entry[idx] << " instead of " << value << " at entry #" << evt_idx + idx << '\n';
         frombuf(tmp_ptr, entry + idx);
         ASSERT_TRUE(compare(entry[idx], value))
            << "In tree " << treename << " Incorrect value for " << branchname
            << " branch: " << entry[idx] << ", expected " << value << " at entry #" << evt_idx + idx
            << (tree->Scan(branchname, "", "", 10, evt_idx + idx - 3), '\n');
         increment( evt_idx + idx, value );
      }
      evt_idx += count;
   }
   sw.Stop();
   EXPECT_EQ(BulkApiMultipleTest::fEventCount, evt_idx);
   printf("GetBulkEntries: Successful read of all events in %s.\n", treename);
   printf("GetBulkEntries: Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
   delete hfile;
}

TEST_F(BulkApiMultipleTest, simpleSerializedReadF)
{
   SimpleSerializedReadFunc<float>(fFileName.c_str(), "T", "myFloat", 2);
}

TEST_F(BulkApiMultipleTest, simpleSerializedReadD)
{
   SimpleSerializedReadFunc<double>(fFileName.c_str(), "T", "myDouble", 3);
}

TEST_F(BulkApiMultipleTest, simpleSerializedReadB)
{
   SimpleSerializedReadFunc<bool>(fFileName.c_str(), "T", "myBool", true);
}

TEST_F(BulkApiMultipleTest, simpleSerializedReadS)
{
   SimpleSerializedReadFunc<short>(fFileName.c_str(), "T", "myShort", 4);
}

TEST_F(BulkApiMultipleTest, simpleSerializedReadI)
{
   SimpleSerializedReadFunc<int>(fFileName.c_str(), "T", "myInt", 5);
}

TEST_F(BulkApiMultipleTest, simpleSerializedReadLL)
{
   SimpleSerializedReadFunc<Long64_t>(fFileName.c_str(), "T", "myLongLong", (1<<20) + 1);
}

TEST_F(BulkApiMultipleTest, simpleSerializedReadC)
{
   SimpleSerializedReadFunc<char>(fFileName.c_str(), "T", "myChar", 7);
}


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
   float idx_f = fFloatInitial;
   double idx_g = fDoubleInitial;
   Int_t events = fEventCount;
   sw.Start();
   while (myReader.Next()) {
      if (R__unlikely(idx == events)) {break;}
      ASSERT_TRUE(compare(*myF, idx_f))
         << "Incorrect value on myFloat branch: " << *myF << ", expected " << idx_f  << " (event " << idx << ")\n";
      ASSERT_TRUE(compare(*myG, idx_g))
         << "Incorrect value on myDouble branch: " << *myG << ", expected " << idx_g  << " (event " << idx << ")\n";
      increment(idx, idx_f);
      increment(idx, idx_g);
      idx++;
   }
   delete hfile;
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
   ASSERT_EQ(ROOT::Internal::TTreeReaderValueBase::kSetupMatch, myF.GetSetupStatus()) <<
      "TTreeReaderValueFast<float> failed to initialize.  Status code: " << myF.GetSetupStatus() << "\n";
   ASSERT_EQ(ROOT::Internal::TTreeReaderValueBase::kSetupMatch, myG.GetSetupStatus()) <<
      "TTreeReaderValueFast<double> failed to initialize.  Status code: " << myG.GetSetupStatus() << "\n";
   ASSERT_EQ(myReader.GetEntryStatus(), TTreeReader::kEntryValid) <<
      "TTreeReaderFast failed to initialize.  Entry status: " <<  myReader.GetEntryStatus() << "\n";
   Int_t events = fEventCount;
   Long64_t idx = 0;
   float idx_f = fFloatInitial;
   double idx_g = fDoubleInitial;
   for (auto reader_idx : myReader) {
      ASSERT_LT(reader_idx, events);
      ASSERT_EQ(reader_idx, idx);
      ASSERT_TRUE(compare(*myF, idx_f))
         << "Incorrect value on myFloat branch: " << *myF << ", expected " << idx_f  << " (event " << idx << ")\n";
      ASSERT_TRUE(compare(*myG, idx_g))
         << "Incorrect value on myDouble branch: " << *myG << ", expected " << idx_g  << " (event " << idx << ")\n";
      increment(idx, idx_f);
      increment(idx, idx_g);
      idx++;
   }
   sw.Stop();
   printf("TTreeReaderFast: Successful read of all events.\n");
   printf("TTreeReaderFast: Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
}
