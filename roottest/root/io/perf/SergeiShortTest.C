#include <iostream>
using namespace std;

#include "TTree.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TBranch.h"
#include "TBenchmark.h"
#include "TSystem.h"
#include "TH1.h"
#include "TH2.h"

void MakeDelay(int time_in_sec) {
   return;
   gSystem->Sleep(time_in_sec*1000);
}

void PurgeMemory(bool show = false) {
   return;

  const int RAMSIZE = 2048;

  if (show) cout << "Purge " << RAMSIZE << " MB of memory" << endl;

  Double_t** buffers = (Double_t**) malloc(RAMSIZE*4);

  for(int n=0;n<RAMSIZE;n++) {
     buffers[n] = new Double_t[125000];
     if (show && (n % 10 == 0)) cout << ".";
     memset(buffers[n],123,1000000);
  }

  if (show) cout << "Done" << endl;

  for(int n=0;n<RAMSIZE;n++)
     delete[] buffers[n];

  free(buffers);
}

void ProduceTree(const char* filename, const char* treename,
                 Int_t numentries, Int_t compression,
                 int numbranches, int branchsize, int buffersize) {
   cout << "Producing tree " << treename << " ..." << endl;

   TFile *f = new TFile(filename,"recreate");
   f->SetCompressionLevel(compression);

   TTree t(treename, treename);
   t.SetMaxTreeSize(19000000000);

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

      t.Branch(brname.Data(),&(data[nbr*branchsize]),format.Data(), buffersize*branchsize*sizeof(Float_t));
   }

   int warn = numentries/10;
   for (int n=0;n<numentries;n++) {
      if (n%warn==0) cout << "Wrote " << n << " entries\n";
      t.Fill();
   }

   t.Write();
   t.GetCurrentFile()->Write();
   delete t.GetCurrentFile();
   delete[] data;
}

void TestTree(const char* filename, const char* treename,
              int numbranches, int branchsize, int activebranches,
              float& RealTime, float& CpuTime) {
   TFile *f = new TFile(filename);
   TTree* t = (TTree*) f->Get(treename);

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

   gBenchmark->Reset();
   gBenchmark->Start("TestTree");

   while (t->GetEntry(counter++));

   gBenchmark->Stop("TestTree");

   gBenchmark->Show("TestTree");

   RealTime = gBenchmark->GetRealTime("TestTree");
   CpuTime = gBenchmark->GetCpuTime("TestTree");

   delete f;
   delete[] data;
}

void RunTest(const char* name, const char* uselesstitle, int numentries, int BufferSize) {
   TString title;
   title += numentries;
   title.Append(" events, 10 branches, 10 floats in brunch, Basket size = ");
   title += ( BufferSize*10*10*4 );
   TH1D* histoRes = new TH1D(name, title, 10, 0.5, 10.5);
   histoRes->GetXaxis()->SetTitle("Number of active branches");
   histoRes->GetYaxis()->SetTitle("Real time (s)");
   histoRes->SetDirectory(0);
   histoRes->SetStats(kFALSE);
   ProduceTree("TreeFile.root","TestTree", numentries, 0, 10, 10, BufferSize);

   Float_t RealTime, CpuTime;

   for(int ActiveBranches=1;ActiveBranches<=10;ActiveBranches++) {
      PurgeMemory();
      cout << "Buffer size = " << BufferSize*sizeof(Float_t)*10 << " ActiveBranches = " << ActiveBranches << endl;
      MakeDelay(5);
      TestTree("TreeFile.root","TestTree", 10, 10, ActiveBranches, RealTime, CpuTime);
      histoRes->SetBinContent(ActiveBranches, RealTime);
   }

   TCanvas* c1 = new TCanvas(TString(name)+"_canvas",title);
   histoRes->Draw();
   c1->SaveAs(TString(name)+".gif");
}


void SergeiShortTest()
{
   RunTest("LargeBuffer","", 40000000, 1000);
    //RunTest("LargeBuffer","1000000 events, 10 branches, 10 floats in brunch, Basket size = 40000", 1000, 1000);
   //RunTest("SmallBuffer","1000000 events, 10 branches, 10 floats in brunch, Basket size = 40", 1000000, 1);
}
