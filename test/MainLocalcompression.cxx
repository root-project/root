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

   TFile *hfile       = 0;
   TTree *tree        = 0;
   TBranch *branch    = 0;
   TLarge *eventlarge = 0;
   TSmall *eventsmall = 0;
   TInt   *eventint   = 0;

   // Fill event, header and tracks with some random numbers
   //   Create a timer object to benchmark this loop
   TStopwatch timer;
   timer.Start();
   Long64_t nb = 0;
   Int_t ev;
   Int_t bufsize;
   Double_t told = 0;
   Double_t tnew = 0;
   Int_t printev = 100;

//         Read case
   if (read) {
      if (object == 1) {
         hfile = new TFile("TSmall.root");
         tree = (TTree*)hfile->Get("T");
         branch = tree->GetBranch("event");
         branch->SetAddress(&eventsmall);
         Int_t nentries = (Int_t)tree->GetEntries();
         nevent = TMath::Min(nevent,nentries); 
      } else if (object == 0) {
         hfile = new TFile("TLarge.root");
         tree = (TTree*)hfile->Get("T");
         branch = tree->GetBranch("event");
         branch->SetAddress(&eventlarge);
         Int_t nentries = (Int_t)tree->GetEntries();
         nevent = TMath::Min(nevent,nentries);
      } else if (object == 2) {
         hfile = new TFile("TInt.root");
         tree = (TTree*)hfile->Get("T");
         branch = tree->GetBranch("event");
         branch->SetAddress(&eventint);
         Int_t nentries = (Int_t)tree->GetEntries();
         nevent = TMath::Min(nevent,nentries);
      }

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
            nb += tree->GetEntry(ev);        //read complete event in memory
         }
      } else {    //read random
         Int_t evrandom = Int_t(nevent*gRandom->Rndm(1));
         nb += tree->GetEntry(evrandom);
      }
   } else {
//         Write case
      // Create a new ROOT binary machine independent file.
      // Note that this file may contain any kind of ROOT objects, histograms,
      // pictures, graphics objects, detector geometries, tracks, events, etc..
      // This file is now becoming the current directory.
    if (object == 1) {
       hfile = new TFile("TSmall.root","RECREATE","TTree benchmark ROOT file");
       eventsmall = new TSmall();
    } else if (object == 0) {
       hfile = new TFile("TLarge.root","RECREATE","TTree benchmark ROOT file");
       eventlarge = new TLarge();
    } else if (object == 2) {
       hfile = new TFile("TInt.root","RECREATE","TTree benchmark ROOT file");
       eventint = new TInt();
    }

    hfile->SetCompressionLevel(comp);

     // Create a ROOT Tree and one superbranch
     tree = new TTree("T","An example of a ROOT tree");
     tree->SetAutoSave(1000000000); // autosave when 1 Gbyte written
     tree->SetCacheSize(10000000);  // set a 10 MBytes cache (useless when writing local files)
     bufsize = 64000;

     TTree::SetBranchStyle(branchStyle);
     if (object == 1) {
        branch = tree->Branch("event", &eventsmall, bufsize,0);
     } else if (object == 0) {
        branch = tree->Branch("event", &eventlarge, bufsize,0);
     } else if (object == 2) {
        branch = tree->Branch("event", &eventint, bufsize, 0);
     }

     branch->SetAutoDelete(kFALSE);
     if(branchStyle) tree->BranchRef();

     for (ev = 0; ev < nevent; ev++) {
        if (ev%printev == 0) {
           tnew = timer.RealTime();
           printf("event:%d, rtime=%f s\n",ev,tnew-told);
           told=tnew;
           timer.Continue();
        }
        if (object == 1) {
           eventsmall->Build();
        } else if (object == 0) {
           eventlarge->Build();
        } else if (object == 2) {
           eventint->Build();
        }
        nb += tree->Fill();  //fill the tree
     }
     hfile = tree->GetCurrentFile(); //just in case we switched to a new file
     hfile->Write();
  }
  // We own the event (since we set the branch address explicitly), we need to delete it.
  if (object == 1) {
     delete eventsmall;  eventsmall = 0;
  } else if (object == 0) {
     delete eventlarge;  eventlarge = 0;
  } else if (object == 2) {
     delete eventint;    eventint = 0;
  }

  //  Stop timer and print results
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
