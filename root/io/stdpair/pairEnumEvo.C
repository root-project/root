#include "TROOT.h"
#include "TClass.h"
#include "TFile.h"
#include <stdlib.h>
#include "TSystem.h"
#include "TVirtualCollectionProxy.h"

// good.C with (0,1,0,0) and good.C(1,0,0,0) behaves differently due to the IsSyntheticPair early return in BuildCheck (not triggered in this case)
// good.C(1,0,0,0): check why order of StreamerInfo is different from (0,1,0,0)

// good.C(0,0,1,0): 'missing' StreamerInfo and after update (use ForceReload in GenCollectionProxy) it now has duplicated StreamerInfo

enum libEnum {
   kShared = 0,
   kACLiC = 1,
   kInterpreted = 2
};

const char *gNameOfPairClass = "pair<reco::Muon::MuonTrackType,edm::Ref<vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<vector<reco::Track>,reco::Track> > >";
const char *gNameOfMap = "map<reco::Muon::MuonTrackType,edm::Ref<vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<vector<reco::Track>,reco::Track> > >";

void printInfo(const char *when)
{
   auto pcl0 = TClass::GetClass(gNameOfPairClass);
   if (pcl0) {
      fprintf(stdout, "\nList of StreamerInfo. %s\n", when);
      fprintf(stdout, "The class has state: %d and classinfo %d\n", pcl0->GetState(), pcl0->HasInterpreterInfo());
      TClass *mcl = TClass::GetClass(gNameOfMap);
      fprintf(stdout, "The map class has state: %d and classinfo %d\n", mcl->GetState(), mcl->HasInterpreterInfo());

      pcl0->GetStreamerInfos()->ls();
   }
}


int pairEnumEvo(int libtype /* 0 shared, 1 ACLiC, 2 interpreted */, bool fixed, bool readbeforeload, bool reportedorder)
{
   // Originally 4 config
   //   load ACLiC or compiled library
   //   with or without p->GetValueClass()
   //gSystem->Load("lib/slc7_amd64_gcc10/libDataFormatsMuonReco.so");
   // + reading the file before the library.
   // + read aa first or ab first

   gInterpreter->SetClassAutoLoading(false);

   if (readbeforeload) {
      TFile *f1 = nullptr;
      TFile *f2 = nullptr;
      if (reportedorder)
         f1 = TFile::Open("aa02.root");
      f2 = TFile::Open("ab02.root");
      if (!reportedorder)
         f1 = TFile::Open("aa02.root");
      // printInfo("After the early file opening.");
      //return 1;
      if (!f1)
         Fatal("pairEnumEvo", "Can not open aa02.root");
      if (!f2)
         Fatal("pairEnumEvo", "Can not open ab02.root");
   }

   if (libtype == libEnum::kACLiC)
      gROOT->ProcessLine(".L cmspair.h+");
   else if (libtype == libEnum::kShared)
      gSystem->Load("libCmsPairCollection.so");
   else if (libtype == libEnum::kInterpreted)
      gROOT->ProcessLine("#include \"cmspair.h\"");
   else {
      fprintf(stderr, "Error: unknown lib value: %d\n",(int)libtype);
      return 1;
   }

   // printInfo("After library loading.");

   auto c = TClass::GetClass(gNameOfMap);
   // printInfo("After map TClass loading.");

   if (!c)
      Fatal("pairEnumEvo", "Not TClass for %s", gNameOfMap);
   c->GetClassInfo();
   // printInfo("After map TClass GetClassInfo.");


   auto p = c->GetCollectionProxy();
   // printInfo("After CollectionProxy loading.");

   if (fixed)
      p->GetValueClass();
 
   if (!readbeforeload) {
      TFile *f1 = nullptr;
      TFile *f2 = nullptr;
      if (reportedorder)
         f1 = TFile::Open("aa02.root");
      f2 = TFile::Open("ab02.root");
      if (!reportedorder)
         f1 = TFile::Open("aa02.root");

      if (!f1)
         Fatal("pairEnumEvo", "Can not open aa02.root");
      if (!f2)
         Fatal("pairEnumEvo", "Can not open ab02.root");
   }

   // printInfo("After the late file opening.");

   // p->GetValueClass()->GetStreamerInfo()->ls();
   // TClass::GetClass(gNameOfPairClass)->GetStreamerInfos()->ls();

   fprintf(stdout, "\nLast verifications:\n");
   fprintf(stdout, "Current StreamerInfo:\n");
   auto pcl = TClass::GetClass(gNameOfPairClass);
   pcl->GetStreamerInfo()->ls();
   auto i = pcl->GetStreamerInfo(2);
   if (i) {
      fprintf(stdout, "\n#2 StreamerInfo:\n");
      i->ls();
   }
   return 0;
}

#if 0

Issues with 'interpreted' + load-file-first mode.

Loading the file creates a TClass for the map.  Additional code (eg accessing
the CollectionProxy) might also create a TClass for the pair.

When the header file is loaded in the interpreter, unless there is explicit
uses of the map (or pair), there is no decl for the instantiation of thus
the TClass for the map and pair are not refreshed.

When the pair or map are instantiated (eg. `gInterpreter->Declare("pair<...> pl")`),
the TClass for the pair is informed (via `TCling::UpdateClassInfoWithDecl`
and `TCling::RefreshClassInfo`).

We could update `RefreshClassInfo` to refresh the `StreamerInfo` for the pair
but it would also need to also refresh the map's CollectionProxy (size,
hints, etc?) [and there is an arbitrary number because they are thread-local]

So at that point, it might actually be better to recreate the TClass for the
map ...

But wait ... there is currently no support for generating a collection proxy for
an interpreted class ... so it is actually an emulated collection proxy ...

That proxy does not match the interpreted (nor the compiled) version of the
map ... so there is no good point to match the pair either ....

So the solution above are (a tad bit) complex and .... not enough ...

#endif

