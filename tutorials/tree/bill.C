// benchmark comparing write/read to/from keys or trees
// for example for N=10000, the following output is produced
// on a P IV 62.4GHz
//
// root -b -q bill.C    or root -b -q bill.C++
//
//billw0  : RT=  1.070 s, Cpu=  1.050 s, File size=  45508003 bytes, CX= 1
//billr0  : RT=  0.740 s, Cpu=  0.730 s
//billtw0 : RT=  0.720 s, Cpu=  0.660 s, File size=  45163959 bytes, CX= 1
//billtr0 : RT=  0.420 s, Cpu=  0.420 s
//billw1  : RT=  6.600 s, Cpu=  6.370 s, File size=  16215349 bytes, CX= 2.80687
//billr1  : RT=  2.290 s, Cpu=  2.270 s
//billtw1 : RT=  3.260 s, Cpu=  3.230 s, File size=   6880273 bytes, CX= 6.5642
//billtr1 : RT=  0.990 s, Cpu=  0.980 s
//billtot : RT= 18.290 s, Cpu= 15.920 s
//******************************************************************
//*  ROOTMARKS = 600.9   *  Root3.05/02   20030201/1840
//******************************************************************
//
//Author: Rene Brun

#include "TFile.h"
#include "TSystem.h"
#include "TH1.h"
#include "TRandom.h"
#include "TStopwatch.h"
#include "TKey.h"
#include "TTree.h"
#include "TROOT.h"

const Int_t N = 10000;       //number of events to be processed
TStopwatch timer;



void billw(const char *billname, Int_t compress) {
   //write N histograms as keys
   timer.Start();
   TFile f(billname,"recreate","bill benchmark with keys",compress);
   TH1F h("h","h",1000,-3,3);
   h.FillRandom("gaus",50000);

   for (Int_t i=0;i<N;i++) {
      char name[20];
      sprintf(name,"h%d",i);
      h.SetName(name);
      h.Fill(2*gRandom->Rndm());
      h.Write();
   }
   timer.Stop();
   printf("billw%d  : RT=%7.3f s, Cpu=%7.3f s, File size= %9d bytes, CX= %g\n",compress,timer.RealTime(),timer.CpuTime(),
          (Int_t)f.GetBytesWritten(),f.GetCompressionFactor());
}

void billr(const char *billname, Int_t compress) {
   //read N histograms from keys
   timer.Start();
   TFile f(billname);
   TIter next(f.GetListOfKeys());
   TH1F *h;
   TH1::AddDirectory(kFALSE);
   TKey *key;
   Int_t i=0;
   TH1F *hmean = new TH1F("hmean","hist mean from keys",100,0,1);

   while ((key=(TKey*)next())) {
      h = (TH1F*)key->ReadObj();
      hmean->Fill(h->GetMean());
      delete h;
      i++;
   }
   timer.Stop();
   printf("billr%d  : RT=%7.3f s, Cpu=%7.3f s\n",compress,timer.RealTime(),timer.CpuTime());
}

void billtw(const char *billtname, Int_t compress) {
   //write N histograms to a Tree
   timer.Start();
   TFile f(billtname,"recreate","bill benchmark with trees",compress);
   TH1F *h = new TH1F("h","h",1000,-3,3);
   h->FillRandom("gaus",50000);
   TTree *T = new TTree("T","test bill");
   T->Branch("event","TH1F",&h,64000,0);
   for (Int_t i=0;i<N;i++) {
      char name[20];
      sprintf(name,"h%d",i);
      h->SetName(name);
      h->Fill(2*gRandom->Rndm());
      T->Fill();
   }
   T->Write();
   delete T;
   timer.Stop();
   printf("billtw%d : RT=%7.3f s, Cpu=%7.3f s, File size= %9d bytes, CX= %g\n",compress,timer.RealTime(),timer.CpuTime(),
                    (Int_t)f.GetBytesWritten(),f.GetCompressionFactor());
}

void billtr(const char *billtname, Int_t compress) {
   //read N histograms from a tree
   timer.Start();
   TFile f(billtname);
   TH1F *h = 0;
   TTree *T = (TTree*)f.Get("T");
   T->SetBranchAddress("event",&h);
   TH1F *hmeant = new TH1F("hmeant","hist mean from tree",100,0,1);
   Long64_t nentries = T->GetEntries();
   for (Long64_t i=0;i<nentries;i++) {
      T->GetEntry(i);
      hmeant->Fill(h->GetMean());
   }
   timer.Stop();
   printf("billtr%d : RT=%7.3f s, Cpu=%7.3f s\n",compress,timer.RealTime(),timer.CpuTime());
}

void bill() {

   TString dir = gSystem->DirName(gSystem->UnixPathName(__FILE__));
   TString bill = dir + "/bill.root";
   TString billt = dir + "/billt.root";

   TStopwatch totaltimer;
   totaltimer.Start();
   for (Int_t compress=0;compress<2;compress++) {
      billw(bill,compress);
      billr(bill,compress);
      billtw(billt,compress);
      billtr(billt,compress);
   }
   gSystem->Unlink(bill);
   gSystem->Unlink(billt);
   totaltimer.Stop();
   Double_t rtime = totaltimer.RealTime();
   Double_t ctime = totaltimer.CpuTime();
   printf("billtot : RT=%7.3f s, Cpu=%7.3f s\n",rtime,ctime);
   //reference is a P IV 2.4 GHz
   Float_t rootmarks = 600*(16.98 + 14.40)/(rtime + ctime);
   printf("******************************************************************\n");
   printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),gROOT->GetVersionDate(),gROOT->GetVersionTime());
   printf("******************************************************************\n");
}

