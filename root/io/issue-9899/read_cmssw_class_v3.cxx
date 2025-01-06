#include "TTree.h"
#include "TFile.h"
#include "TError.h"
#include <vector>

struct RefVectorMemberPointersHolder {
   std::vector<long> fTransient; 
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
      fTransient.resize(s);
      fHolder.resize(s);
   }
   size_t size() {
      return fHolder.size() + fTransient.size();
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
   long fPx = 0;
};

#ifdef __ROOTCLING__
#pragma read sourceClass="RefVectorBase" targetClass="RefVectorBase" source="" \
        versions="1-" target="fTransient" code="{ fTransient.clear(); }"
#pragma read sourceClass="RefVectorMemberPointersHolder" targetClass="RefVectorMemberPointersHolder" source="" \
        versions="1-" target="fTransient" code="{ fTransient.clear(); }"
#pragma read sourceClass="LowData" targetClass="LowData" source="Values fValues" \
        checksum="[0x16066232]" target="fNewValues" code="{ fNewValues = onfile.fValues; }"
#endif

struct LowData {
   long fLong = 0;
   float fFloat = 0;
   char fPadding[256];  //!
   RefVector fProblems;
   RefVector fLoose;
   RefVector fTight;
   RefVector fNoise;
   Values fNewValues;
   void resize(size_t s) {
      fProblems.resize(s);
      fLoose.resize(s+1);
      fTight.resize(s+2);
      fNoise.resize(s+3);
   }
   int check(int slide = 1, size_t expected = 0) {
      int result = 0;
      bool expectempty = (expected == 0);
      result += slide * (fProblems.size()/2 != expected);
      result += slide * BIT(1) * (fLoose.size()/2 != (expected + (!expectempty) * 1));
      result += slide * BIT(2) * (fTight.size()/2 != (expected + (!expectempty) * 2));
      result += slide * BIT(3) * (fNoise.size()/2 != (expected + (!expectempty) * 3));
      if (expectempty)
         result += slide * BIT(4) * ( fNewValues.fPx != 2);
      return result;
   }
};

struct Fine {
   float fA;
   float fB;
};

struct HighData {
   std::vector<Fine> fFine;
   std::vector<LowData> fLowData;
   std::vector<LowData> fPersLowData;

   void resize(size_t s) {
      fFine.resize(s);
      fPersLowData.resize(s);
      for(auto &d : fPersLowData)
         d.resize(s);
      fLowData.resize(s);
      for(auto &d : fLowData)
         d.resize(s);
   }
   int check(int slide = 1) {
      int res = 0;
      for(auto &d : fPersLowData) {
         res += d.check(slide, 10);
      }
      for(auto &d : fLowData)
         res += d.check(slide * BIT(6), 10);
      return res;
   }
};
struct PersHighData {
   std::vector<Fine> fFine;
   std::vector<LowData> fPersLowData;  //
   std::vector<LowData> fLowData;      // 'new' in CMSSSW has 2 of this.

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
   int check(int slide = 1) {
      int res = 0;
#ifdef FIXED_ISSUE_9924
      for(auto &d : fPersLowData)
         res += d.check(slide, 0);
#endif
      for(auto &d : fLowData)
         res += d.check(slide * BIT(6), 10);
      return res;

   }
};

struct Wrapper {
   HighData fValue;
   std::vector<HighData> fVecValues;   //
   PersHighData fPersValue;
};

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
   int final = 0;
   for(Long64_t e = 0; e < t->GetEntriesFast(); ++e) {
      if (w) {
         for(auto &d : w->fVecValues) 
            d.resize(10);
         w->fValue.resize(10);
         w->fPersValue.resize(10);
      }
      t->GetEntry(e);
      if (w) {
         int res = 0;
         for (auto &d : w->fVecValues) {
            res += d.check();
         }
         res += w->fValue.check();
         res += w->fPersValue.check(BIT(7));
         if (res) {
            Error("readfile", "For entries %lld we have result=%d", e, res);
         }
         final += res;
      }
   }
   return final; 
}

int read_cmssw_class_v3() { return readfile(); }
