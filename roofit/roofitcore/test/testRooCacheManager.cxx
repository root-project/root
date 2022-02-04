// Tests for the RooCacheManager
// Author: Jonas Rembser, CERN, May 2021

#include "RooAbsPdf.h"
#include "RooGenericPdf.h"
#include "RooObjCacheManager.h"
#include "RooRealVar.h"

#include "TFile.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

TEST(RooCacheManager, TestSelectFromArgSet)
{
   // Test RooCacheManager::selectFromSet1 and RooCacheManager::selectFromSet2.

   // the cached class doesn't matter for this test, it just has to be an object that the cache is going to own
   class CacheElem : public RooAbsCacheElement {
   public:
      ~CacheElem() override {}
      RooArgList containedArgs(Action) override { return {}; }
   };

   // RooObjCacheManager is a wrapper around RooCacheManager
   RooObjCacheManager mgr{nullptr, 100};

   // fill some vector with RooAbsReals for testing
   std::vector<RooRealVar> vars;
   for (int i = 0; i < 10; ++i) {
      auto name = std::string("v") + std::to_string(i);
      vars.emplace_back(name.c_str(), name.c_str(), static_cast<double>(i));
   }

   RooArgSet nset1{vars[0], vars[1], "nset1"};
   RooArgSet iset1{vars[2], vars[3], vars[4], "nset2"};

   RooArgSet nset2{vars[5], vars[6], vars[7], "nset1"};
   RooArgSet iset2{vars[8], vars[9], "nset2"};

   int idx1 = mgr.setObj(&nset1, &iset1, new CacheElem);
   int idx2 = mgr.setObj(&nset2, &iset2, new CacheElem);

   auto sel11 = mgr.selectFromSet1({vars[0], vars[4]}, idx1);
   auto sel12 = mgr.selectFromSet2({vars[2], vars[4]}, idx1);

   auto sel21 = mgr.selectFromSet1({vars[1], vars[6]}, idx2);
   auto sel22 = mgr.selectFromSet2({vars[0], vars[1]}, idx2);

   // check if the expected number of args were selected
   EXPECT_EQ(sel11.size(), 1);
   EXPECT_EQ(sel12.size(), 2);
   EXPECT_EQ(sel21.size(), 1);
   EXPECT_EQ(sel22.size(), 0);

   // check if the correct args were selected
   EXPECT_TRUE(sel11.containsInstance(vars[0]));
   EXPECT_TRUE(sel12.containsInstance(vars[2]));
   EXPECT_TRUE(sel12.containsInstance(vars[4]));
   EXPECT_TRUE(sel21.containsInstance(vars[6]));
}

TEST(RooCacheManager, TestAbsArgCacheListConsistency)
{
   // Every instance of a RooAbsCache or inherigin class that is the member of
   // a RooFit arg is automatically added to the RooAbsArg::_cacheList data
   // member by reference.
   //
   // This test makes sure that the _cacheList still has the correct pointers
   // after reading back a RooFit model. Now that the RooAbsCache and child
   // classes don't take part in the IO anymore it should be no problem, but in
   // the past there were inconsistencies.
   {
      RooRealVar x("x", "x", 0, -10, 10);
      RooGenericPdf pdf("pdf", "pdf", "x", RooArgList(x));

      uintptr_t pdfAddr = (uintptr_t)&pdf;
      uintptr_t cacheAddr = (uintptr_t)pdf.getCache(0);
      // Check if the cache pointer actually points to a data member of the pdf.
      // We can't get the actual address of the private _normMgr member, but we
      // can still check if the cache points to a member by checking the range
      // of the address.
      EXPECT_TRUE(cacheAddr >= pdfAddr && cacheAddr < pdfAddr + sizeof(RooGenericPdf));
      EXPECT_EQ(pdf.numCaches(), 1);

      TFile f1("testRooCacheManager_1.root", "RECREATE");

      x.Write();
      pdf.Write();
   }

   {
      TFile f1("testRooCacheManager_1.root", "READ");
      auto pdf = f1.Get<RooAbsPdf>("pdf");

      uintptr_t pdfAddr = (uintptr_t)pdf;
      uintptr_t cacheAddr = (uintptr_t)pdf->getCache(0);
      // Same trick to check if the cache points to a pdf data member as above.
      EXPECT_TRUE(cacheAddr >= pdfAddr && cacheAddr < pdfAddr + sizeof(RooGenericPdf));
      EXPECT_EQ(pdf->numCaches(), 1);
   }
}
