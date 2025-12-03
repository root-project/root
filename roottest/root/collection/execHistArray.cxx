// This test checks the I/O backward compatibility of classes TArray(C|S|I|L|L64|F|D)
//    after the introduction of the template class TArrayT
#include <cassert>
#include <cmath>
#include <limits>
#include <cstring>
#include "Riostream.h"
#include "TClass.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1C.h"
#include "TH2D.h"
#include "TH2I.h"
#include "TH3S.h"
#include "TArrayI.h"
#include "TArrayC.h"
#include "TArrayS.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "TArrayL.h"
#include "TArrayL64.h"

void write(const char *filename = "HistArray.root")
{
   TFile* f = TFile::Open(filename, "RECREATE");

   // Since the goal is to the test the backward compatibility of TArray, we are calling every
   // TArray function in the test and building all possible TArray derived types. Moreover,
   // we are creating and checking histograms, which are derived from TArray classes.


   // 1. DIRECT TESTING OF TArray CLASSES

   TArrayC* tc = new TArrayC(20);
   for(Int_t i = 0; i < 20; ++i) (*tc)[i] = i;
   f->WriteObjectAny(tc, "TArrayC", "tc");

   TArrayS* ts = new TArrayS();
   f->WriteObjectAny(ts, "TArrayS", "ts");

   TArrayI* ti = new TArrayI(12);
   ti->Reset(47);
   f->WriteObjectAny(ti, "TArrayI", "ti");

   TArrayL* tl = new TArrayL(10);
   Long_t al[5] = { 1000L, 10000L, 100000L, 1000000L, 2000000L };
   tl->Adopt(5, al);
   f->WriteObjectAny(tl, "TArrayL", "tl");

   Long64_t al64[3] = { 4000000L, 7000000L, 8000000L };
   TArrayL64* tl64 = new TArrayL64(3, al64);
   f->WriteObjectAny(tl64, "TArrayL64", "tl64");

   TArrayF* tf = new TArrayF(2);
   tf->SetAt(35.24f, 0);
   tf->SetAt(-198.5239f, 1);
   TArrayF* tfcopy = new TArrayF(*tf);
   f->WriteObjectAny(tf, "TArrayF", "tf");
   f->WriteObjectAny(tfcopy, "TArrayF", "tfcopy");

   TArrayD* td = new TArrayD(1);
   td->AddAt(123.562798, 0);
   f->WriteObjectAny(td, "TArrayD", "td");



   // 2. INDIRECT TESTING OF TArray through histograms (TH)

   TH1* hc = new TH1C("hc", "hc", 100, 0, 20);
   hc->SetBinContent(100, 45);
   hc->SetBinContent(1, -20);
   hc->AddBinContent(2);
   hc->Write();

   TH1* hf = new TH1F("hf", "hf", 10, 0, 10);
   hf->SetBinContent(9, 5.62f);
   hf->AddBinContent(10, -0.35f);
   hf->Write();

   TH2* hd = new TH2D("hd", "hd", 30, -2, 5, 10, 0, 1);
   hd->SetBinContent(300, 1e9);
   hd->Write();

   TH3* hs = new TH3S("hs", "hs", 100, 0, 4, 35, -2, 4, 15, -6, 3);
   hs->SetBinContent(0, 12); hs->AddBinContent(0, -35);
   hs->Write();

   TH2* hi = new TH2I("hi", "hi", 46, 0, 6, 14, -10, -9);
   hi->AddBinContent(46, -2);
   hi->AddBinContent(14,  3);
   hi->AddBinContent(12, 11); hi->AddBinContent(12, -3);
   hi->Write();

   f->Close();
}


void read(const char *filename = "HistArray.root")
{
   TFile* f = TFile::Open(filename, "READ");

   // 1. DIRECT TESTING OF TArray CLASSES

   // Checking correct construction and content
   TArrayC* tc; f->GetObject("tc", tc);
   assert(tc != NULL); assert(std::strcmp(tc->IsA()->GetName(), "TArrayC") == 0);
   assert(tc->GetSize() == 20);
   for(Int_t i = 0; i < tc->GetSize(); ++i) {
      assert(tc->GetAt(i) == i);
      assert(tc->GetAt(i) == tc->At(i));
   }
   assert(tc->GetSum() == 190); // 0 + 1 + 2 + ... + 19

   // Checking correct construction of an empty TArray
   TArrayS* ts; f->GetObject("ts", ts);
   assert(ts != NULL); assert(std::strcmp(ts->IsA()->GetName(), "TArrayS") == 0);
   assert(ts->GetArray() == NULL);
   assert(ts->GetSize() == 0);
   assert(ts->GetSum() == 0);

   // Checking Reset function
   TArrayI* ti; f->GetObject("ti", ti);
   assert(ti != NULL); assert(std::strcmp(ti->IsA()->GetName(), "TArrayI") == 0);
   assert(ti->GetSize() == 12);
   for(Int_t i = 0; i < ti->GetSize(); ++i) assert(ti->GetAt(i) == 47);

   // Checking Adopt function
   TArrayL* tl; f->GetObject("tl", tl);
   assert(tl != NULL); assert(std::strcmp(tl->IsA()->GetName(), "TArrayL") == 0);
   assert(tl->GetSize() == 5);
   assert((*tl)[0] == 1000L);
   assert((*tl)[1] == 10000L);
   assert((*tl)[2] == 100000L);
   assert((*tl)[3] == 1000000L);
   assert((*tl)[4] == 2000000L);

   // Checking raw array constructor (and associated Set function)
   TArrayL64* tl64; f->GetObject("tl64", tl64);
   assert(tl64 != NULL); assert(std::strcmp(tl64->IsA()->GetName(), "TArrayL64") == 0);
   assert(tl64->GetSize() == 3);
   assert((*tl64)[0] == 4000000L);
   assert((*tl64)[1] == 7000000L);
   assert((*tl64)[2] == 8000000L);

   // Checking copy constructor
   TArrayF* tf    ; f->GetObject("tf"    , tf    );
   TArrayF* tfcopy; f->GetObject("tfcopy", tfcopy);
   assert(tf     != NULL); assert(std::strcmp(tf->IsA()->GetName()    , "TArrayF") == 0);
   assert(tfcopy != NULL); assert(std::strcmp(tfcopy->IsA()->GetName(), "TArrayF") == 0);
   assert(tf->GetSize() == tfcopy->GetSize());
   assert(std::abs(tf->GetSum() - tfcopy->GetSum()) < std::numeric_limits<Float_t>::epsilon());
   for(Int_t i = 0; i < tf->GetSize(); ++i)
      assert(std::abs(tf->At(i) - tfcopy->At(i)) < std::numeric_limits<Float_t>::epsilon());

   // Checking AddAt function
   TArrayD* td; f->GetObject("td", td);
   assert(td != NULL); assert(std::strcmp(td->IsA()->GetName(), "TArrayD") == 0);
   assert(td->GetSize() == 1);
   assert(std::abs(td->At(0) - 123.562798) < std::numeric_limits<Double_t>::epsilon());



   // 2. INDIRECT TESTING OF TArray through histograms (TH)

   TH1C* hc; f->GetObject("hc", hc);
   assert(hc != NULL); assert(std::strcmp(hc->IsA()->GetName(), "TH1C") == 0);
   assert(hc->GetBinContent(100) == 45);
   assert(hc->GetBinContent(1) == -20);
   assert(hc->GetBinContent(2) == 1);
   for(Int_t i = 3; i < 100; ++i) assert(hc->GetBinContent(i) == 0);

   TH1F* hf; f->GetObject("hf", hf);
   assert(hf != NULL); assert(std::strcmp(hf->IsA()->GetName(), "TH1F") == 0);
   assert(std::abs(hf->GetBinContent(9)  - 5.62f) < std::numeric_limits<Float_t>::epsilon());
   assert(std::abs(hf->GetBinContent(10) + 0.35f) < std::numeric_limits<Float_t>::epsilon());
   for(Int_t i = 0; i < 9; ++i) assert(hf->GetBinContent(i) < std::numeric_limits<Float_t>::epsilon());

   TH2D* hd; f->GetObject("hd", hd);
   assert(hd != NULL); assert(std::strcmp(hd->IsA()->GetName(), "TH2D") == 0);
   assert(std::abs(hd->GetBinContent(300) - 1e9) < std::numeric_limits<Double_t>::epsilon());
   assert(std::abs(hd->GetAt(300) - 1e9) < std::numeric_limits<Double_t>::epsilon());

   TH3S* hs; f->GetObject("hs", hs);
   assert(hs != NULL); assert(std::strcmp(hs->IsA()->GetName(), "TH3S") == 0);
   assert(hs->GetBinContent(0) == -23);
   assert(hs->GetSum() == -23);

   TH2I* hi; f->GetObject("hi", hi);
   assert(hi != NULL); assert(std::strcmp(hi->IsA()->GetName(), "TH2I") == 0);
   assert(hi->GetSum() == 9);

   f->Close();
}


int execHistArray(const char* filename = "HistArray.root")
{
   read(filename);
   return 0;
}




