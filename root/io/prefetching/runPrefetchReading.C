#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include "TSystem.h"
#include "TTreePerfStats.h"
#include "TVirtualStreamerInfo.h"
#include "TStreamerElement.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TEnv.h"
#include "TFileCacheRead.h"
#include "TTreeCache.h"
#include "TError.h"

Int_t runPrefetchReading()
{
  //const char *options = 0;
   Int_t freq = 1000;
   Int_t cachesize = -1;
   Float_t percententries = 1.00;
   Float_t percentbranches = 1.00;
   TStopwatch sw;
   
   //set the reading mode to async prefetching
   gEnv->SetValue("TFile.AsyncPrefetching", 1);
   
   // open the local if any
   TString filename("atlasFlushed.root");
   if (gSystem->AccessPathName(filename,kReadPermission) && filename.Index(":") == kNPOS) {
      // otherwise open the http file
      filename.Prepend("http://root.cern.ch/files/");
      //filename.Prepend("root://cache01.usatlas.bnl.gov//data/test1/");
   }

   TFile *file = TFile::Open( filename );
   if (!file || file->IsZombie()) return 1;

   // Try the known names :)
   const char *names [] = { "E","Events","CollectionTree","ntuple","T" };
   TTree *T = NULL;
   for (unsigned int i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
      file->GetObject(names[i], T);
      if (T) break;
   }
   if (T==0) {
     Error("runPrefetchReading","Could not find a tree which the conventional names in %s.",filename.Data());
     return 2;
   }
   TFile::SetReadaheadSize(0);  // (256*1024);
   Long64_t nentries = T->GetEntries();

   int efirst = 0;
   int elast  = efirst+nentries;
   if (cachesize == -2) {
      gEnv->SetValue("TFile.AsyncReading", 0);
      cachesize = -1;
   }
   T->SetCacheSize(cachesize);

   if (cachesize != 0) {
      T->SetCacheEntryRange(efirst,elast);
      if (percentbranches < 1.00) {
         int nb = T->GetListOfBranches()->GetEntries();
         int incr = nb * percentbranches;
         for(int b=0;b < nb; b += incr) T->AddBranchToCache(((TBranch*)T->GetListOfBranches()->At(b)),kTRUE);
      } else {
         T->AddBranchToCache("*");
      }
      T->StopCacheLearningPhase();
   }
  
   TRandom r;
   for (Long64_t i=efirst;i<elast;i++) {
     if (i % freq == 0){
       // for (Long64_t i=elast-1;i>=efirst;i--) {
       if (i%freq == 0) printf("i = %lld\n",i);
       if (r.Rndm() > percententries) continue; 
       T->LoadTree(i);
       if (percentbranches < 1.00) {
         int nb = T->GetListOfBranches()->GetEntries();
         int incr = nb * percentbranches;
         for(int b=0;b<nb; b += incr) ((TBranch*)T->GetListOfBranches()->At(b))->GetEntry(i);   
         int count = 0;
         int maxcount = 100 + 100 ;
         for(int x = 0; x < maxcount; ++x ) { /* waste cpu */ count = sin(cos((double)count)); }
       } else {
         T->GetEntry(i);
       }
     }
   }
 
   fprintf(stdout, "fPrefetchedBlocks = %lli\n", file->GetCacheRead()->GetPrefetchedBlocks());
   return 0;
}
