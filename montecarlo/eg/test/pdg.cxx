// Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
// All rights reserved.
//
// For the licensing terms see $ROOTSYS/LICENSE.
// For the list of contributors see $ROOTSYS/README/CREDITS.

// @author Danilo Piparo CERN

#include <TDatabasePDG.h>
#include <TInterpreter.h>
#ifdef R__USE_IMT
#include <ROOT/TThreadExecutor.hxx>
#endif
#include <TROOT.h>

#include "gtest/gtest.h"

#include <thread>
#include <vector>

#ifdef R__USE_IMT
constexpr unsigned int NUM_THREADS = 8u;

void RunOnThreads(std::function<void(void)> &&func)
{
   ROOT::EnableThreadSafety();
   std::vector<std::thread> v;
   for (unsigned int i = 0; i < NUM_THREADS; ++i) {
      v.emplace_back(std::thread(func));
   }
   for (auto &t : v)
      t.join();
}

TEST(TDatabasePDGMT, Initialization)
{
   RunOnThreads(TDatabasePDG::Instance);
   delete TDatabasePDG::Instance();
}

void CheckPi0(TParticlePDG &pi0)
{
   EXPECT_TRUE(0 == pi0.Stable());
   EXPECT_EQ(0, pi0.Beauty());
   EXPECT_EQ(0, pi0.Charge());
   EXPECT_EQ(0, pi0.Charm());
   EXPECT_EQ(0, pi0.Isospin());
   EXPECT_EQ(0, pi0.Strangeness());
   EXPECT_EQ(0, pi0.Top());
   EXPECT_DOUBLE_EQ(1.349768e-01, pi0.Mass());
   EXPECT_DOUBLE_EQ(7.810000e-09, pi0.Width());
   EXPECT_NEAR(8.4278090781e-17, pi0.Lifetime(),0.0000000001e-17);

}

TEST(TDatabasePDGMT, GetParticleByName)
{
   auto f = []() { 
      auto pi0 = TDatabasePDG::Instance()->GetParticle("pi0"); 
      CheckPi0(*pi0);};
   RunOnThreads(f);
   delete TDatabasePDG::Instance();
}

TEST(TDatabasePDGMT, GetParticleByCode)
{
   auto f = []() { 
      auto pi0 = TDatabasePDG::Instance()->GetParticle(111); 
      CheckPi0(*pi0);};
   RunOnThreads(f);
   delete TDatabasePDG::Instance();
}

TEST(TDatabasePDGMT, GetParticleByCodeAndName)
{
   auto f = []() {
      const auto code = TDatabasePDG::Instance()->GetParticle("pi0")->PdgCode();
      auto pi0 = TDatabasePDG::Instance()->GetParticle(code);
      CheckPi0(*pi0);
   };
   RunOnThreads(f);
   delete TDatabasePDG::Instance();
}

#endif

// ROOT-6889
TEST(TDatabasePDG, AntiParticleStability)
{
   auto db = TDatabasePDG::Instance();
   // W- is unstable
   EXPECT_FALSE(db->GetParticle(-24)->Stable());
}
