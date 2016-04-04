// @(#)root/test:$Id$
// Author: Rene Brun   19/01/97

#include <stdlib.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TFile.h"
#include "TNetFile.h"
#include "TRandom.h"
#include "TTree.h"
#include "TBranch.h"
#include "TClonesArray.h"
#include "TStopwatch.h"

#include "Localcompression.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
   Int_t nevent = 40;     // by default create 400 events
   Int_t comp   = 1;       // by default file is compressed
   Int_t read   = 0;
   Int_t object  = 0;
   Int_t rac = 0;

   if (argc > 1)  nevent = atoi(argv[1]);
   if (argc > 2)  comp   = atoi(argv[2]);
   if (argc > 3)  read   = atoi(argv[3]);
   if (argc > 4)  object  = atoi(argv[4]);
   if (argc > 5)  rac = atoi(argv[5]);

   if (rac) gROOT->SetRandomAccessCompression(1);
   else gROOT->SetRandomAccessCompression(0);

   Int_t branchStyle = 1; //new style by default

   TFile *hfile;
   TTree *tree;
   TLarge *eventlarge = 0;
   TSmall *eventsmall = 0;
   TInt   *eventint   = 0;
   Int_t nsmall = 1000;
   Int_t nint = nsmall*100;

   TStopwatch timer;
   timer.Start();
   Long64_t nb = 0;
   Long64_t nbl = 0;
   Long64_t nbs = 0;
   Long64_t nbi = 0;

   Int_t ev;
   Int_t bufsize;
   Double_t told = 0;
   Double_t tnew = 0;
   Int_t printev = 100;

   TBranch *branch;
   TBranch *branchsmall;
   TBranch *branchlarge;
   TBranch *branchint;

// Read case
   if (read) {
      if (rac)
         hfile = new TFile("TCombineRAC.root");
      else
         hfile = new TFile("TCombineNOR.root");
      tree = (TTree*)hfile->Get("T");
      if (object == 1) {
         branch = tree->GetBranch("EventSmall");
         branch->SetAddress(&eventsmall);
      } else if (object == 0) {
         branch = tree->GetBranch("EventLarge");
         branch->SetAddress(&eventlarge);
      } else if (object == 2) {
         branch = tree->GetBranch("EventInt");
         branch->SetAddress(&eventint);
      } else {
         branch = 0;
      }

      Int_t nentries = (Int_t)branch->GetEntries();
      nevent = TMath::Min(nevent,nentries); 
      printf("nentries=%d,nevent=%d\n",nentries,nevent);
      if (read == 1) {  //read sequential
         //by setting the read cache to -1 we set it to the AutoFlush value when writing
         Int_t cachesize = -1;
         tree->SetCacheSize(cachesize);
         tree->SetCacheLearnEntries(1); //one entry is sufficient to learn
         tree->SetCacheEntryRange(0,nevent);
         for (ev = 0; ev < nevent; ev++) {
            tree->LoadTree(ev);  //this call is required when using the cache
            if (ev%printev == 0) {
               tnew = timer.RealTime();
               printf("event:%d, rtime=%f s\n",ev,tnew-told);
               told=tnew;
               timer.Continue();
            }
            nb += branch->GetEntry(ev);        //read branch in memory
         }
      } else {    //read random
         Int_t distance = nentries/nevent;
         Int_t cnt = 0;
         for (ev = 0; ev < nentries; ev+=distance) {
            tree->LoadTree(ev);
            cnt++;
            if (cnt%printev == 0) {
               tnew = timer.RealTime();
               printf("event:%d, rtime=%f s\n",ev,tnew-told);
               told=tnew;
               timer.Continue();
            }
            nb += branch->GetEntry(ev);
         }
      }
   } else {
//  Write case
      if (rac)
         hfile = new TFile("TCombineRAC.root","RECREATE","TTree benchmark ROOT file");
      else
         hfile = new TFile("TCombineNOR.root","RECREATE","TTree benchmark ROOT file");

      eventsmall = new TSmall();
      eventlarge = new TLarge();
      eventint = new TInt();

      hfile->SetCompressionLevel(comp);

      // Create a ROOT Tree and one superbranch
      tree = new TTree("T","An example of a ROOT tree");
      tree->SetAutoSave(1000000000); // autosave when 1 Gbyte written
      tree->SetCacheSize(10000000);  // set a 10 MBytes cache (useless when writing local files)
      bufsize = 64000;

      TTree::SetBranchStyle(branchStyle);
      branchsmall = tree->Branch("EventSmall", &eventsmall, bufsize,0);
      branchlarge = tree->Branch("EventLarge", &eventlarge, bufsize,0);
      branchint   = tree->Branch("EventInt", &eventint, bufsize, 0);

      branchsmall->SetAutoDelete(kFALSE);
      branchlarge->SetAutoDelete(kFALSE);
      branchint->SetAutoDelete(kFALSE);

      if(branchStyle) tree->BranchRef();

      for (ev = 0; ev < nevent; ev++) {
         if (ev%printev == 0) {
            tnew = timer.RealTime();
            printf("event:%d, rtime=%f s\n",ev,tnew-told);
            told=tnew;
            timer.Continue();
         }
         eventlarge->Build();
         nbl += branchlarge->Fill();
//         printf("event%d, nb large = %lld\n", ev, nbl);
         for (Int_t i = 0; i < nsmall; ++i) {
            eventsmall->Build();
            nbs += branchsmall->Fill();
//            printf("event%d, iner%d, nb small = %lld\n", ev, i, nbs);
         }
         for (Int_t i = 0; i < nint; ++i) {
            eventint->Build();
            nbi += branchint->Fill();
//            printf("event%d, iner%d, nb int = %lld\n", ev, i, nbi);
         }
//         nb += tree->Fill();  //fill the tree
//         printf("nb total = %lld\n", nb);
      }
      nb = nbl + nbs + nbi;
      hfile = tree->GetCurrentFile(); //just in case we switched to a new file
      hfile->Write();
   }

   // We own the event (since we set the branch address explicitly), we need to delete it.
   delete eventsmall;  eventsmall = 0;
   delete eventlarge;  eventlarge = 0;
   delete eventint;    eventint = 0;

   // Stop timer and print results
   timer.Stop();
   Float_t mbytes = 0.000001*nb;
   Double_t rtime = timer.RealTime();
   Double_t ctime = timer.CpuTime();

   printf("\n%d events and %lld bytes processed.\n",nevent,nb);
   printf("RealTime=%f seconds, CpuTime=%f seconds\n",rtime,ctime);

   if (read) {
      tree->PrintCacheStats();
      printf("You read %f Mbytes/Realtime seconds\n",mbytes/rtime);
      printf("You read %f Mbytes/Cputime seconds\n",mbytes/ctime);
   } else {
      printf("compression level=%d\n", comp);
      printf("You write %f Mbytes/Realtime seconds\n",mbytes/rtime);
      printf("You write %f Mbytes/Cputime seconds\n",mbytes/ctime);
      //printf("file compression factor = %f\n",hfile.GetCompressionFactor());
   }
   hfile->Close();
   return 0;
}
