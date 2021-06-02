// Tests for the RooCacheManager
// Author: Jonas Rembser, CERN, May 2021

#include "RooObjCacheManager.h"
#include "RooRealVar.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

TEST(RooCacheManager, TestSelectFromArgSet)
{
   // Test RooCacheManager::selectFromSet1 and RooCacheManager::selectFromSet2.

   // the cached class doesn't matter for this test, it just has to be an object that the cache is going to own
   class CacheElem : public RooAbsCacheElement {
   public:
      virtual ~CacheElem() {}
      virtual RooArgList containedArgs(Action) { return {}; }
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

   std::cout << idx1 << std::endl;
   std::cout << idx2 << std::endl;

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
