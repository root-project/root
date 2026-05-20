#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
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
#include "TSystemDirectory.h"

#include <cmath>

Int_t runPrefetchReading(bool prefetch = true, bool caching = false)
{
   //const char *options = 0;
   Int_t freq = 1000;
   Int_t cachesize = -1;
   Float_t percententries = 1.00;
   Float_t percentbranches = 1.00;
   TStopwatch sw;

   //set the reading mode to async prefetching
   if (prefetch) gEnv->SetValue("TFile.AsyncPrefetching", 1);

   //enable the local caching of blocks
   TString cachedir="file:/tmp/xcache/";
   // or using xrootd on port 2000
   // TString cachedir="root://localhost:2000//tmp/xrdcache1/";
   if (caching) gEnv->SetValue("Cache.Directory", cachedir.Data());

   // open the local if any
   TString filename("atlasFlushed.root");
   if (gSystem->AccessPathName(filename,kReadPermission) && filename.Index(":") == kNPOS) {
      // otherwise open the http file
      filename.Prepend("root://eospublic.cern.ch//eos/root-eos/testfiles/");
      //filename.Prepend("http://root.cern.ch/files/");
      //filename.Prepend("root://cache01.usatlas.bnl.gov//data/test1/");
      //filename.Prepend( "root://pcitdss1401//tmp/" );
      //filename.Prepend("http://www-root.fnal.gov/files/");
      //filename.Prepend("http://oink.fnal.gov/distro/roottest/");
   } else {
      fprintf(stderr,"Using local file\n");
   }

   fprintf(stderr,"Starting to load the library\n");
   gSystem->Load("libRoottestIoPrefetching");

   fprintf(stderr,"Starting to open the file\n");
   TFile *file = TFile::Open( filename, "TIMEOUT=30" );
   if (!file || file->IsZombie()) {
      Error("runPrefetchReading","Could not open the file %s within 30s",filename.Data());
      return 1;
   }
   fprintf(stderr,"The file has been opened, setting up the TTree\n");

   // file->MakeProject("atlasFlushed","*","RECREATE+");

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

   //auto ps = new TTreePerfStats("stats", T);

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

   //...........................................................................
   // First read, with saving the info in cache
   //...........................................................................
   fprintf(stderr,"Setup done. Starting to read the entries\n");
   TRandom r;
   for (Long64_t i = efirst; i < elast; i++) {
     //if (i%100 == 0 || i>2000) fprintf(stderr,"i.debug = %lld\n",i);
     // if (i==2000) gDebug = 7;
     if (i % freq == 0){
       // for (Long64_t i=elast-1;i>=efirst;i--) {
       if (i%freq == 0 || i==(elast-1)) fprintf(stderr,"i = %lld\n",i);
       if (r.Rndm() > percententries) continue;
       T->LoadTree(i);
       if (percentbranches < 1.00) {
         int nb = T->GetListOfBranches()->GetEntries();
         int incr = nb * percentbranches;
         for(int b=0;b<nb; b += incr) ((TBranch*)T->GetListOfBranches()->At(b))->GetEntry(i);
         int count = 0;
         int maxcount = 1000 + 100 ;
         for(int x = 0; x < maxcount; ++x ) { /* waste cpu */ count = sin(cos((double)count)); }
       } else {
         T->GetEntry(i);
       }
     }
   }

   //ps->Print("basket");
   //ps->SaveAs("treestats.root");

   fprintf(stderr,"Done reading for the first pass, now closing the file\n");
   file->Close();
   delete file;

   //...........................................................................
   // Second read, actually reading the data from cache
   //...........................................................................
   fprintf(stderr,"Opening the file for the 2nd pass\n");
   file = TFile::Open( filename, "TIMEOUT=30" );
   if (!file || file->IsZombie()) return 1;

   fprintf(stderr,"The file has been opened, setting up the TTree\n");
   // Try the known names :)
   for (unsigned int i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
      file->GetObject(names[i], T);
      if (T) break;
   }
   if (T==0) {
     Error("runPrefetchReading","Could not find a tree which the conventional names in %s.",filename.Data());
     return 2;
   }

   TFile::SetReadaheadSize(0);  // (256*1024);
   nentries = T->GetEntries();

   efirst = 0;
   elast  = efirst+nentries;

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

   fprintf(stderr,"Setup done, starting the 2nd reading.\n");
   for (Long64_t i = efirst; i < elast; i++) {
     if (i % freq == 0){
       // for (Long64_t i=elast-1;i>=efirst;i--) {
       if (i%freq == 0 || i==(elast-1)) fprintf(stderr,"i = %lld\n",i);
       if (r.Rndm() > percententries) continue;
       T->LoadTree(i);
       if (percentbranches < 1.00) {
         int nb = T->GetListOfBranches()->GetEntries();
         int incr = nb * percentbranches;

         for(int b=0;b<nb; b += incr) {
           ((TBranch*)T->GetListOfBranches()->At(b))->GetEntry(i);
         }

         int count = 0;
         int maxcount = 1000 + 100 ;
         for(int x = 0; x < maxcount; ++x ) {
           /* waste cpu */
           count = sin(cos((double)count));
         }
       } else {
         T->GetEntry(i);
       }
     }
   }
   fprintf(stderr, "Done with the 2nd reading\n");

   fprintf(stderr, "fPrefetchedBlocks = %lli\n", file->GetCacheRead()->GetPrefetchedBlocks());

   fprintf(stderr, "Delete tmp directory: /tmp/xcache\n" );
   if (caching) {
      gSystem->Unlink("/tmp/xcache");
   }

   file->Close();
   delete file;
   return 0;
}
