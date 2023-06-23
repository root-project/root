#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1D.h>
#include <TInterpreter.h>

#include "gtest/gtest.h"

// ROOT-10023
TEST(TTreeReaderBasic, BranchObject)
{
   TTree t("t", "t");
   TObject *o = nullptr;
   t.Branch("o", &o, 32000, 0); // must be unsplit to generate a TBranchObject

   // Fill branch with different concrete types
   TNamed name("name", "title");
   TList list;
   list.Add(&name);
   o = &list;
   t.Fill();
   TH1D h("h", "h", 100, 0, 100);
   h.Fill(42);
   o = &h;
   t.Fill();
   o = nullptr;

   TTreeReader r(&t);
   TTreeReaderValue<TObject> rv(r, "o");
   EXPECT_TRUE(r.Next());
   EXPECT_STREQ(rv->ClassName(), "TList");
   auto *asList = static_cast<TList *>(rv.Get());
   EXPECT_EQ(asList->GetEntries(), 1);
   EXPECT_STREQ(asList->At(0)->GetTitle(), "title");
   EXPECT_STREQ(asList->At(0)->GetName(), "name");

   EXPECT_TRUE(r.Next());
   EXPECT_STREQ(rv->ClassName(), "TH1D");
   EXPECT_DOUBLE_EQ(static_cast<TH1D *>(rv.Get())->GetMean(), 42.);
}

// Issue #12334
// branchobject.root created as
//
// ~~~ {.cpp}
// f = new TFile("branchobject.root", "RECREATE");
// TTree *t = new TTree("branchobject", "test tree for branchobject.cxx");
// ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t>> lv(1., 2., 3., 4.);
// t->Branch("lv32", &lv);
// t->Fill();
// t->Write();
// ~~~
TEST(TTreeReaderBasic, LorentzVector32)
{
   // Ensure that the mismatching `<double>` specialization is available, i.e. will
   // be chosen given the typeid of the TTreeReaderValue template argument.
   TClass::GetClass("ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >");

   TFile file("branchobject.root");
   TTreeReader reader("branchobject");
   std::string code;
   {
      std::stringstream sstr;
      sstr << "TTreeReaderValue<ROOT::Math::PtEtaPhiMVector> lv32(*(TTreeReader*)"
         << &reader << ", \"lv32\");";
      code = sstr.str();
   }
   gInterpreter->Declare(code.c_str());
   ASSERT_TRUE(reader.Next());
   EXPECT_EQ(gInterpreter->Calc("int(lv32->pt() + 0.5)"), 1);
   EXPECT_EQ(gInterpreter->Calc("int(lv32->eta() + 0.5)"), 2);
   EXPECT_EQ(gInterpreter->Calc("int(lv32->phi() + 0.5)"), 3);
   EXPECT_EQ(gInterpreter->Calc("int(lv32->M() + 0.5)"), 4);
}
