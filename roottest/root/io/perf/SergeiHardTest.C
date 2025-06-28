#include <iostream.h>
#include "TTree.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TBranch.h"
#include "TBenchmark.h"
#include "TSystem.h"
#include "TH1.h"
#include "TH2.h"

void MakeDelay(int time_in_sec) {
   gSystem->Sleep(time_in_sec*1000);
}

void ProduceTree(const char* filename, const char* treename,
                 Int_t numentries, Int_t compression,
                 int numbranches, int branchsize, int buffersize) {
   TFile f(filename,"RECREATE");
   f.SetCompressionLevel(compression);

   TTree* t = new TTree(treename, treename);
   t->SetMaxTreeSize(19000000000.);

   Float_t* data = new Float_t[numbranches * branchsize];
   for (int i=0;i<numbranches * branchsize;i++)
      data[i] = 1.2837645786 * i;

   for (int nbr=0;nbr<numbranches;nbr++) {
      TString brname = "Branch";
      brname+=nbr;

      TString format = brname;
      format+="[";
      format+=branchsize;
      format+="]/F";

      t->Branch(brname.Data(),&(data[nbr*branchsize]),format.Data(), buffersize*branchsize*sizeof(Float_t));
   }

   for (int n=0;n<numentries;n++)
     t->Fill();

   t->Write();

   delete t;

   delete[] data;
}

void TestTree(const char* filename, const char* treename,
              int numbranches, int branchsize, int activebranches,
              float* RealTime, float* CpuTime) {
   TFile f(filename);
   TTree* t = (TTree*) f.Get(treename);

   Float_t* data = new Float_t[numbranches*branchsize];

   for(int nbr=0;nbr<numbranches;nbr++) {
      TString brname = "Branch";
      brname+=nbr;
      t->SetBranchAddress(brname,&(data[nbr*branchsize]));
   }

   if (activebranches<=0) t->SetBranchStatus("*",1); else {
      t->SetBranchStatus("*",0);
      for (int nbr=0;nbr<activebranches;nbr++) {
         TString brname = "Branch";
         brname+=nbr;
         t->SetBranchStatus(brname,1);
      }
   }

   Int_t counter = 0;

   if (RealTime && CpuTime) {
      gBenchmark->Reset();
      gBenchmark->Start("TestTree");
   }

   while (t->GetEntry(counter++));

   if (RealTime && CpuTime) {
      gBenchmark->Stop("TestTree");

      gBenchmark->Show("TestTree");

      *RealTime = gBenchmark->GetRealTime("TestTree");
      *CpuTime = gBenchmark->GetCpuTime("TestTree");
   }

   delete[] data;
}

void DoDummy() {
   const int RAMSIZE = 2048;

   // file size is double of memory size
  int numentries = RAMSIZE * 2 * 2750;

  cout << "Clean up disk cash ..." << endl;
  // ProduceTree("DummyFile.root","DummyTree", numentries, 0, 10, 10, 10000);
  TestTree("DummyFile.root","DummyTree", 10, 10, 10, 0, 0);
}

void RunTest(const char* name, int numentries, int BufferSize) {
   char title[200];
   sprintf(title, "%d events, 10 branches, 10 floats in brunch, Basket size = %d", numentries, BufferSize*sizeof(Float_t)*10);

   TH1D* histoRes = new TH1D(name, title, 10, 0.5, 10.5);
   histoRes->GetXaxis()->SetTitle("Number of active branches");
   histoRes->GetYaxis()->SetTitle("Real time (s)");
   histoRes->SetDirectory(0);
   histoRes->SetStats(kFALSE);

   cout << "Producing tree ..." << endl;

   ProduceTree("TreeFile.root","TestTree", numentries, 0, 10, 10, BufferSize);

   Float_t RealTime, CpuTime;

   for(int ActiveBranches=10;ActiveBranches>0;ActiveBranches--) {
      DoDummy();
      cout << "Buffer size = " << BufferSize*sizeof(Float_t)*10 << " ActiveBranches = " << ActiveBranches << endl;
      MakeDelay(5);
      TestTree("TreeFile.root","TestTree", 10, 10, ActiveBranches, &RealTime, &CpuTime);
      histoRes->SetBinContent(ActiveBranches, RealTime);
   }

   TCanvas* c1 = new TCanvas(TString(name)+"_canvas", title);
   histoRes->Draw();
   c1->SaveAs(TString(name)+".gif");
}


void SergeiHardTest(bool more = false)
{
   RunTest("Test_1M_100K",   1000000, 2500);
   RunTest("Test_1M_1M",     1000000, 25000);
   if(more) {
      RunTest("Test_1M_200K",   1000000, 5000);
      RunTest("Test_1M_50K",    1000000, 1250);
      RunTest("Test_1M_1K",     1000000, 25);
      RunTest("Test_1M_10K",    1000000, 250);
      RunTest("Test_1M_400K",   1000000, 10000);
   }

   //   RunTest("Second","4000000 events, 10 branches, 10 floats in brunch, Basket size = 40000", 4000000, 1000);
  //   RunTest("SmallBuffer","1000000 events, 10 branches, 10 floats in brunch, Basket size = 40", 1000000, 1);

}
