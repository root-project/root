#include "TROOT.h"
#include "TClass.h"
#include "TFile.h"
#include <stdlib.h>
#include "TSystem.h"
#include "TVirtualCollectionProxy.h"
#include "TError.h"

// good.C with (0,1,0,0) and good.C(1,0,0,0) behaves differently due to the IsSyntheticPair early return in BuildCheck (not triggered in this case)
// good.C(1,0,0,0): check why order of StreamerInfo is different from (0,1,0,0)

// good.C(0,0,1,0): 'missing' StreamerInfo and after update (use ForceReload in GenCollectionProxy) it now has duplicated StreamerInfo

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

void printElements(TVirtualStreamerInfo *info)
{
   if (info->GetElements()) {
      TIter    next(info->GetElements());
      while (TNamed *obj = (TNamed*)next())
         if (0 != strcmp("Emulation", obj->GetTitle()))
            obj->SetTitle("");
   }

   info->ls();
}

enum libEnum {
   kShared = 0,
   kACLiC = 1,
   kInterpreted = 2,
   kPairShared = 3,
   kNothing = 4
};

int pairEnumEvo(int libtype /* used as enum libEnum */, bool fixed, bool readbeforeload, bool reportedorder)
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
      if (!f1)
         Fatal("pairEnumEvo", "Can not open aa02.root");
      if (!f2)
         Fatal("pairEnumEvo", "Can not open ab02.root");
   }

   if (libtype == libEnum::kACLiC)
      gROOT->ProcessLine(".L cmspair.h+");
   else if (libtype == libEnum::kShared)
      gSystem->Load("libCmsPairCollection.so");
   else if (libtype == libEnum::kPairShared)
      gSystem->Load("libCmsPair.so");
   else if (libtype == libEnum::kInterpreted)
      gROOT->ProcessLine("#include \"cmspair.h\"");
   else if (libtype == libEnum::kNothing)
      {} // Do nothing.
   else {
      fprintf(stderr, "Error: unknown lib value: %d\n",(int)libtype);
      return 1;
   }

   // printInfo("After library loading.");

   auto c = TClass::GetClass(gNameOfMap);

   if (!c)
      Fatal("pairEnumEvo", "Not TClass for %s", gNameOfMap);
   c->GetClassInfo();


   auto p = c->GetCollectionProxy();

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

   fprintf(stdout, "\nLast verifications:\n");
   fprintf(stdout, "Current StreamerInfo:\n");
   auto pcl = TClass::GetClass(gNameOfPairClass);
   auto currentInfo = pcl->GetStreamerInfo();
   printElements(currentInfo);
   auto ninfos = pcl->GetStreamerInfos()->GetSize() - 1;
   for(int i = 1; i < ninfos ; ++i) {
      auto info = dynamic_cast<TStreamerInfo*>(pcl->GetStreamerInfo(i));
      if (info && info != currentInfo) {
         fprintf(stdout, "\n#%d StreamerInfo:\n", i);
         printElements(info);
      }
   }
   // pcl->GetStreamerInfos()->ls();
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

