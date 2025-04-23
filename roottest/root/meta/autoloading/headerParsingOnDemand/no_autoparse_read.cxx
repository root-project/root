#include "TError.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include <algorithm>

void TestErrorHandler(Int_t level, Bool_t abort_bool, const char *location, const char *msg);

auto gMainHandler = SetErrorHandler(TestErrorHandler);


void CheckForProceed(const char *msg)
{
   static const char *target = "We can proceed for ";
   static auto size = strlen(target);
   if (0 == strncmp(msg, target, size)) {
      Fatal("AutoParsePreventionTester", "Actual autoparse requested for: %s", msg + size);
   }
}

void TestErrorHandler(Int_t level, Bool_t abort_bool, const char *location, const char *msg)
{ 
   if (level > kInfo) {
      gMainHandler(level, abort_bool, location, msg);
   } else if (0 == strcmp("TCling::AutoParse", location) || 0 == strcmp("TInterpreter::TCling::AutoParse", location)) {
      CheckForProceed(msg);
// Test problems
/*
edm::Value, reco::Muon, edmNew::dstvdetails::DetSetVectorTrans
*/
// CMS actual problems:
/*
Info in <TInterpreter::TCling::AutoParse>: We can proceed for edm::Hash<1>. We have 1 headers.
Info in <TInterpreter::TCling::AutoParse>: We can proceed for edm::ParameterSetBlob. We have 1 headers.
Info in <TInterpreter::TCling::AutoParse>: We can proceed for edm::Hash. We have 1 headers.
Info in <TInterpreter::TCling::AutoParse>: We can proceed for L1CSCSPStatusDigi. We have 1 headers.
Info in <TInterpreter::TCling::AutoParse>: We can proceed for reco::Muon. We have 2 headers.
*/
      gMainHandler(level, abort_bool, location, msg);
   } else {
      // be quiet
   }
}

int no_autoparse_read(const char *filename = "no_autoparse_v10.root", Long64_t nentries = 10)
{
   if (gSystem->Load("libno_autoparse_v11") != 0)
      Fatal("no_autoparse_read", "Could not load the library libno_autoparse_v11");
   // The gDebug level 2 will cause some printout that are currently done with straight
   // printf, so we just essentially ignore it.
   gInterpreter->ProcessLine(".1> no_autoparse_read.stdout.log");
   gDebug = 2;
   TFile *f = TFile::Open(filename);
   if (!f || f->IsZombie())
      Fatal("no_autoparse_read", "Could not open the file %s", filename);
   TTree *t = f->Get<TTree>("Events");
   if (!t)
      Fatal("no_autoparse_read", "Could not find the TTree");
   nentries = std::min(t->GetEntries(), nentries);

   for(long long e = 0; e < nentries; ++e)
      t->GetEntry(e);

   gDebug = 0;
   delete f;
   return 0;   	
}

