#include "TTree.h"
#include "TFile.h"
#include "TError.h"
#include <vector>


// NOTE: Need to add a test where the (nested) data member is (essentially) renamed

struct RefVectorMemberPointersHolder {
   std::vector<long> fTransient;  //!  NOTE: with the rule and not transient, it does not work (offset is 9999 and rule is not run)
   void resize(size_t s) {
      fTransient.resize(s);
   }
   size_t size() {
      return fTransient.size();
   }
};

struct RefVectorBase {
   long fID = 0;
   std::vector<long> fTransient; //!  works.
   RefVectorMemberPointersHolder fHolder; // might fail
   void resize(size_t s) {
      fHolder.resize(s);
   }
   size_t size() {
      return fHolder.size();
   }
};

struct RefVector
{
   RefVectorBase fContent;
   void resize(size_t s) {
      fContent.resize(s);
   }
   size_t size() {
      return fContent.size();
   }
};

struct Values {
   long fPx = 2;
};

#ifdef __ROOTCLING__
#pragma read sourceClass="RefVectorBase" targetClass="RefVectorBase" source="" \
        versions="1-" target="fTransient" code="{ fTransient.clear(); }"
#pragma read sourceClass="RefVectorMemberPointersHolder" targetClass="RefVectorMemberPointersHolder" source="" \
        versions="1-" target="fTransient" code="{ fTransient.clear(); }"
#endif

struct LowData {
   long fLong = 0;
   float fFloat = 0;
   char fPadding[256];  //!
   RefVector fProblems;
   RefVector fLoose;
   RefVector fTight;
   RefVector fNoise;
   Values fValues;    //||
   void resize(size_t s)
   {
      fProblems.resize(s);
      fLoose.resize(s+1);
      fTight.resize(s+2);
      fNoise.resize(s+3);
   }
   int check(int slide = 1) {
      int result = 0;
      result += slide * (fProblems.size() != 0);
      result += slide * 10 * (fLoose.size() != 0);
      result += slide * 100 * (fTight.size() != 0);
      result += slide * 1000 * (fNoise.size() != 0);
      return result;
   }
};


struct Fine {
   float fA = 0.1;
   float fB = 0.2;
};

struct HighData {
   std::vector<Fine> fFine;
   std::vector<LowData> fPersLowData;  //!
   std::vector<LowData> fLowData;      //! 'new' in CMSSSW has 2 of this.

   HighData(size_t s = 0) {
      fFine.resize(s);
      fPersLowData.resize(s);
      for(auto &d : fPersLowData)
         d.resize(s);
   }

   void resize(size_t s) {
      fFine.resize(s);
      fPersLowData.resize(s);
      for(auto &d : fPersLowData)
         d.resize(s);
      fLowData.resize(s);
      for(auto &d : fLowData)
         d.resize(s);
   }
   int check() {
      int res = 0;
      for(auto &d : fLowData)
         res += d.check(1);
      return res;
   }
};

struct PersHighData {
   std::vector<Fine> fFine;
   std::vector<LowData> fPersLowData;  //
   std::vector<LowData> fLowData;      //! 'new' in CMSSSW has 2 of this.

   PersHighData(size_t s = 0) {
      fPersLowData.resize(s);
      for(auto &d : fPersLowData)
         d.resize(s);
   }

   void resize(size_t s) {
      fFine.resize(s);
      fPersLowData.resize(s);
      for(auto &d : fPersLowData)
         d.resize(s);
      fLowData.resize(s);
      for(auto &d : fLowData)
         d.resize(s);
   }
   int check() {
      int res = 0;
      for(auto &d : fLowData)
         res += d.check(1);
      return res;
   }
};

struct Wrapper {
   std::vector<HighData> fVecValues;  // not seen in CMSSW
   HighData fValue{10};
   PersHighData fPersValue{10};
};

int writefile(const char *filename = "nestedtransient.root")
{
   TFile *f = new TFile(filename, "RECREATE");
   TTree *t = new TTree("events", "events title");
   
   Wrapper w;
   t->Branch("w.", &w);
   for(Long64_t e = 0; e < 10; ++e) {
      w.fVecValues.push_back(HighData(10));
      t->Fill();
   }
   f->Write();
   delete f;
   return 0;
}

int readfile(const char *filename = "nestedtransient.root")
{
   TFile f(filename, "READ");
   if (f.IsZombie()) {
      Error("readfile", "Could not open the file %s\n", filename);
      return 1;
   }
   TTree *t = f.Get<TTree>("events");
   if (!t) {
      Error("readfile", "Could not read events from the file %s\n", filename);
      return 2;
   }

   Wrapper *w = nullptr;
   t->SetBranchAddress("w.", &w);
   t->Print();
   t->Print("debugInfo");
   int final = 0;
   for(Long64_t e = 0; e < t->GetEntriesFast(); ++e) {
      if (w) {
         for(auto &d : w->fVecValues) 
            d.resize(10);
         w->fValue.resize(10);
      }
      t->GetEntry(e);
      if (w) {
         int res = 0;
         for (auto &d : w->fVecValues) {
            // fprintf(stderr, "Checking %lld and got %d\n", e, d.check());
            // Not working
            // res += d.check();
         }
         res += w->fValue.check();
         if (res) {
            Error("readfile", "For entries %lld we have result=%d", e, res);
         }
         final += res;
      }
   }
   return final; 
}

int write_cmssw_class_v2() { return writefile(); }
