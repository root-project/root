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

TEST(TDatabasePDGMT, GetParticleByName)
{
   RunOnThreads([]() { TDatabasePDG::Instance()->GetParticle("pi0"); });
   delete TDatabasePDG::Instance();
}

TEST(TDatabasePDGMT, GetParticleByCode)
{
   RunOnThreads([]() { TDatabasePDG::Instance()->GetParticle(111); });
   delete TDatabasePDG::Instance();
}

TEST(TDatabasePDGMT, GetParticleByCodeAndName)
{
   auto f = []() {
      const auto code = TDatabasePDG::Instance()->GetParticle("pi0")->PdgCode();
      TDatabasePDG::Instance()->GetParticle(code);
   };
   RunOnThreads(f);
   delete TDatabasePDG::Instance();
}

#endif
