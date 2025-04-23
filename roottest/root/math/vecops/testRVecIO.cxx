#include <ROOT/RVec.hxx>
#include <TFile.h>
#include <TInterpreter.h>
#include <TKey.h>
#include <TSystem.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>

#include <algorithm> // std::equal
#include <array>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

class RVecIO : public testing::Test {
public:
   // Called before the first test is run
   static void SetUpTestSuite()
   {
      TFile f(RVecIO::fname, "recreate");
      TTree t("t", "t");
      ROOT::RVec<bool> vb = boolVals[0];
      ROOT::RVec<int> vi = intVals[0];
      t.Branch("vb", &vb);
      t.Branch("vi", &vi);
      t.Fill();
      vb = boolVals[1];
      vi = intVals[1];
      t.Fill();
      t.Write();
   }

   // Called after the last test is run
   static void TearDownTestSuite() { gSystem->Unlink(fname); }

   static const char *fname;
   static const std::array<std::vector<bool>, 2> boolVals;
   static const std::array<std::vector<int>, 2> intVals;
};

const char *RVecIO::fname = "testRVecIO_input.root";
const std::array<std::vector<bool>, 2> RVecIO::boolVals = {{{true, false, true}, {true, false, true, false, true}}};
const std::array<std::vector<int>, 2> RVecIO::intVals = {{{1, 2, 3}, {42, 42, 42, 42, 42, 42, 42}}};

TEST_F(RVecIO, ReadAsStdVector)
{
   TFile f(RVecIO::fname);
   auto t = f.Get<TTree>("t");
   auto vb = new std::vector<bool>();
   auto vi = new std::vector<int>();
   t->SetBranchAddress("vb", &vb);
   t->SetBranchAddress("vi", &vi);
   for (auto e = 0ll; e < 2ll; ++e) {
      ASSERT_GE(t->GetEntry(e), 0);
      EXPECT_EQ(*vb, RVecIO::boolVals[e]);
      EXPECT_EQ(*vi, RVecIO::intVals[e]);
   }
}

TEST_F(RVecIO, ReadWithTTree)
{
   TFile f(RVecIO::fname);
   auto t = f.Get<TTree>("t");
   auto vb = new ROOT::RVec<bool>();
   auto vi = new ROOT::RVec<int>();
   t->SetBranchAddress("vb", &vb);
   t->SetBranchAddress("vi", &vi);
   for (auto e = 0ll; e < 2ll; ++e) {
      ASSERT_GE(t->GetEntry(e), 0);
      EXPECT_TRUE(std::equal(vb->begin(), vb->end(), RVecIO::boolVals[e].begin()));
      EXPECT_TRUE(std::equal(vi->begin(), vi->end(), RVecIO::intVals[e].begin()));
   }
}

TEST_F(RVecIO, ReadWithTTreeReaderArray)
{
   TFile f(RVecIO::fname);
   TTreeReader r("t", &f);
   TTreeReaderArray<bool> rb(r, "vb");
   TTreeReaderArray<int> ri(r, "vi");
   for (int i = 0; i < 2; ++i) {
      ASSERT_TRUE(r.Next());
      EXPECT_EQ(rb.GetSize(), RVecIO::boolVals[i].size());
      EXPECT_TRUE(std::equal(rb.begin(), rb.end(), RVecIO::boolVals[i].begin()));
      EXPECT_EQ(ri.GetSize(), RVecIO::intVals[i].size());
      EXPECT_TRUE(std::equal(ri.begin(), ri.end(), RVecIO::intVals[i].begin()));
   }
}

// This is similar to what happens in RDataframe
TEST_F(RVecIO, AdoptReaderArrayMemory)
{
   TFile f(RVecIO::fname);
   TTreeReader r("t", &f);
   TTreeReaderArray<int> ri(r, "vi");

   ASSERT_TRUE(r.Next());
   auto riSize = ri.GetSize();
   EXPECT_EQ(riSize, RVecIO::intVals[0].size());
   auto riAddr = &ri.At(0);
   ROOT::RVec<int> rveci(riAddr, riSize); // memory adoption
   EXPECT_EQ(rveci.size(), RVecIO::intVals[0].size());
   EXPECT_TRUE(std::equal(rveci.begin(), rveci.end(), RVecIO::intVals[0].begin()));

   ASSERT_TRUE(r.Next());
   riSize = ri.GetSize();
   EXPECT_EQ(riSize, RVecIO::intVals[1].size());
   riAddr = &ri.At(0);
   ROOT::RVec<int> rveci2(riAddr, riSize);
   std::swap(rveci, rveci2);
   EXPECT_EQ(rveci.size(), RVecIO::intVals[1].size());
   EXPECT_TRUE(std::equal(rveci.begin(), rveci.end(), RVecIO::intVals[1].begin()));
}

TEST(RVecWriteObject, WriteAndReadObjectAny)
{
   const auto fname = "rvecio_writeobject.root";
   {
      TFile f(fname, "recreate");
      ROOT::RVec<int> vi{1, 2, 3};
      ROOT::RVec<bool> vb{true, false, true};

      f.WriteObjectAny(&vi, "ROOT::RVec<int>", "vi");
      f.WriteObjectAny(&vb, "ROOT::RVec<bool>", "vb");
      f.Close();
   }

   TFile f(fname);
   auto *vi = static_cast<ROOT::RVec<int>*>(f.GetKey("vi")->ReadObjectAny(TClass::GetClass("ROOT::RVec<int>")));
   auto *vb = static_cast<ROOT::RVec<bool>*>(f.GetKey("vb")->ReadObjectAny(TClass::GetClass("ROOT::RVec<bool>")));
   ASSERT_TRUE(vi != nullptr);
   ASSERT_TRUE(vb != nullptr);

   EXPECT_TRUE(All(*vi == ROOT::RVec<int>({1, 2, 3})));
   EXPECT_TRUE(All(*vb == ROOT::RVec<bool>({true, false, true})));

   gSystem->Unlink(fname);
}
