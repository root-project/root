// benchmark comparing write/read to/from keys or trees
// for example for N=10000, the following output is produced
// on a P III 600 Mhz
   
// root -b -q bill.C    or root -b -q bill.C++
//
// billw0  : RT=  3.030 s, Cpu=  3.010 s, File size=  45507955 bytes, CX= 1
// billr0  : RT=  4.110 s, Cpu=  4.110 s
// billtw0 : RT=  2.160 s, Cpu=  2.160 s, File size=  45163899 bytes, CX= 1
// billtr0 : RT=  2.040 s, Cpu=  2.040 s
// billw1  : RT= 17.420 s, Cpu= 17.420 s, File size=  16215301 bytes, CX= 2.80687
// billr1  : RT=  7.640 s, Cpu=  7.640 s
// billtw1 : RT=  8.690 s, Cpu=  8.690 s, File size=   6884429 bytes, CX= 6.56023
// billtr1 : RT=  3.040 s, Cpu=  3.040 s
// billtot : RT= 62.330 s, Cpu= 49.210 s
// ******************************************************************
// *  ROOTMARKS = 200.6   *  Root3.03/07   20020810/1539
// ******************************************************************

#include "TFile.h"
#include "TSystem.h"
#include "TH1.h"
#include "TRandom.h"
#include "TStopwatch.h"
#include "TKey.h"
#include "TTree.h"
   
const Int_t N = 10000;       //number of events to be processed
TStopwatch timer;

void billw(Int_t compress) {
   //write N histograms as keys
   timer.Start();
   TFile f("/tmp/bill.root","recreate","bill benchmark with keys",compress);
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
void billr(Int_t compress) {
   //read N histograms from keys
   timer.Start();
   TFile f("/tmp/bill.root");
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
void billtw(Int_t compress) {
   //write N histograms to a Tree
   timer.Start();
   TFile f("/tmp/billt.root","recreate","bill benchmark with trees",compress);
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
void billtr(Int_t compress) {
   //read N histograms from a tree
   timer.Start();
   TFile f("/tmp/billt.root");
   TH1F *h = 0;
   TTree *T = (TTree*)f.Get("T");
   T->SetBranchAddress("event",&h);
   TH1F *hmeant = new TH1F("hmeant","hist mean from tree",100,0,1);
   Int_t nentries = (Int_t)T->GetEntries();
   for (Int_t i=0;i<nentries;i++) {
      T->GetEntry(i);
      hmeant->Fill(h->GetMean());
   }
   timer.Stop();
   printf("billtr%d : RT=%7.3f s, Cpu=%7.3f s\n",compress,timer.RealTime(),timer.CpuTime());
}
void bill() {
   
   TStopwatch totaltimer;
   totaltimer.Start();
   for (Int_t compress=0;compress<2;compress++) {
      billw(compress);
      billr(compress);
      billtw(compress);
      billtr(compress);
   }
   gSystem->Unlink("/tmp/bill.root");
   gSystem->Unlink("/tmp/billt.root");
   totaltimer.Stop();
   Double_t rtime = totaltimer.RealTime();
   Double_t ctime = totaltimer.CpuTime();
   printf("billtot : RT=%7.3f s, Cpu=%7.3f s\n",rtime,ctime);
   //reference is a P III 600 Mhz
   Float_t rootmarks = 200*(62.3 + 49.21)/(rtime + ctime);
   printf("******************************************************************\n");
   printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),gROOT->GetVersionDate(),gROOT->GetVersionTime());
   printf("******************************************************************\n");
}
          
